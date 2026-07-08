"""수동 하이라이트 편집기 / Manual highlight builder.

이미 받아둔(로컬) 영상 파일과 사용자가 직접 입력한 하이라이트 시간대만으로
요약 영상을 만든다. 다운로드/오디오 에너지 분석 단계를 건너뛰고,
summarizer.py 의 검증된 cut_and_concat() 을 그대로 재사용한다.

시간대 입력 형식(한 줄에 하나):
    1:23 - 2:05
    83 - 125
    00:01:23,000 --> 00:02:05,000     (SRT 스타일도 허용)
구분자는 '-', '~', '->' , '-->' 모두 허용. 시각은 SS / MM:SS / HH:MM:SS.
"""
import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

# 포터블(임베디드) 파이썬은 python311._pth 때문에 스크립트 폴더를 sys.path 에
# 자동 추가하지 않는다. 같은 폴더의 summarizer 등을 import 하려면 직접 넣어준다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from summarizer import (
    TRANSITION_STYLES,
    SFX_SPECS,
    WM_POSITIONS,
    GAME_PROMPT,
    build_chapters,
    cut_and_concat,
    compute_punchin_times,
    apply_overlays,
    extract_audio,
    transcribe,
    build_srt,
    get_duration,
    safe_filename,
    silence_cut,
    set_hw_encoding,
)


def parse_time(token: str) -> float:
    """'1:23' / '01:02:03' / '83' / '83.5' / '00:01:23,500' -> 초(float)."""
    token = token.strip().replace(",", ".")
    if not token:
        raise ValueError("빈 시간 값")
    parts = token.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"시간 형식 오류: {token}")


_SEP_RE = re.compile(r"\s*(?:-->|->|~|-|–|—|to)\s*", re.IGNORECASE)


def _split_label(line: str) -> Tuple[str, str]:
    """'1:23 - 2:05 | 다운그레이드' -> ('1:23 - 2:05', '다운그레이드').
    '|' 뒤가 없으면 소제목은 빈 문자열."""
    if "|" in line:
        time_part, label = line.split("|", 1)
        return time_part.strip(), label.strip()
    return line.strip(), ""


def parse_labeled_ranges(text: str) -> List[Tuple[float, float, str]]:
    """여러 줄의 시간대 텍스트를 (start, end, label) 리스트로 파싱.
    각 줄은 'start - end' 또는 'start - end | 소제목' 형식."""
    ranges: List[Tuple[float, float, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        time_part, label = _split_label(line)
        # 콤마로 start,end 를 준 경우도 허용
        if _SEP_RE.search(time_part):
            a, b = _SEP_RE.split(time_part, maxsplit=1)
        elif "," in time_part and time_part.count(",") == 1:
            a, b = time_part.split(",", 1)
        else:
            raise ValueError(f"시작-끝 구분자를 찾을 수 없습니다: '{line}'")
        start = parse_time(a)
        end = parse_time(b)
        if end <= start:
            raise ValueError(f"끝 시간이 시작 시간보다 빨라요: '{line}'")
        ranges.append((start, end, label))
    return ranges


def parse_ranges(text: str) -> List[Tuple[float, float]]:
    """여러 줄의 시간대 텍스트를 (start, end) 리스트로 파싱(소제목은 무시)."""
    return [(s, e) for s, e, _ in parse_labeled_ranges(text)]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="로컬 영상 + 수동 하이라이트 시간대 -> 요약 영상")
    parser.add_argument("video", help="로컬 영상 파일 경로")
    parser.add_argument("--ranges", default="",
                        help="하이라이트 시간대(여러 줄). 각 줄 'start - end'. "
                             "미지정 시 --ranges-file 사용")
    parser.add_argument("--ranges-file", default="",
                        help="하이라이트 시간대를 담은 텍스트 파일 경로")
    parser.add_argument("--output-dir", default="output", help="출력 폴더 (기본: output)")
    parser.add_argument("--name", default="",
                        help="출력 파일 이름(확장자 제외). 미지정 시 원본 파일명 사용")
    parser.add_argument("--transition-style", default="black",
                        choices=list(TRANSITION_STYLES.keys()),
                        help="화면 전환: none / black / white / closeup (기본 black)")
    parser.add_argument("--cam-region", default="br",
                        help="캠 위치 (closeup/punchin 모드): tl/tr/bl/br 또는 x,y,w,h (기본 br)")
    parser.add_argument("--closeup-every", type=int, default=1,
                        help="클로즈업을 몇 번의 전환마다 넣을지 (1=매번, 2=2회당 1회, 3=3회당 1회). 기본 1")
    parser.add_argument("--closeup-sec", type=float, default=1.5,
                        help="클로즈업/펀치인 길이(초). 기본 1.5")
    parser.add_argument("--sfx", dest="sfx_kind", default="whoosh",
                        choices=list(SFX_SPECS.keys()),
                        help="전환 효과음 (기본 whoosh)")
    parser.add_argument("--no-transition", action="store_true",
                        help="화면 전환/효과음 모두 끄기")
    parser.add_argument("--punchin", default="none",
                        choices=["none", "low", "mid", "high"],
                        help="구간 중간 캠 강조(펀치인): none(끔) / low(적게) / mid(보통) / high(많이). 기본 none")
    parser.add_argument("--punchin-times", default="",
                        help="펀치인 시간 직접 지정 (선택). 예: 12:30, 45:02, 1:03:11 (쉼표 구분, 원본 영상 기준). "
                             "자동(--punchin level)과 병합됨 - 같은 구간에서 3초 이내 중복은 제거")
    parser.add_argument("--subtitles", action="store_true",
                        help="완성 영상에서 자막(SRT) 자동 생성 (Whisper)")
    parser.add_argument("--model", default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper 모델 (자막 켤 때만 사용, 기본 small)")
    parser.add_argument("--lang", default="ko", help="자막 언어 코드 (기본 ko)")
    parser.add_argument("--prompt", default=GAME_PROMPT,
                        help="Whisper initial_prompt (전문 용어 힌트)")
    parser.add_argument("--label-pos", default="tr", choices=list(WM_POSITIONS.keys()),
                        help="하이라이트 소제목 위치: tl(좌상) tr(우상) bl(좌하) br(우하). 기본 tr")
    parser.add_argument("--label-size", type=int, default=40,
                        help="하이라이트 소제목 글자 크기 (기본 40)")
    parser.add_argument("--font", default="Paperlogy",
                        help="소제목 글꼴 (기본 Paperlogy)")
    parser.add_argument("--jump-cut", action="store_true",
                        help="무음 구간 자동 컷(점프컷)으로 템포를 높입니다")
    parser.add_argument("--cpu-encode", action="store_true",
                        help="GPU 가속 인코딩 끄기 (호환성 문제 시)")
    args = parser.parse_args()

    if args.cpu_encode:
        from summarizer import set_hw_encoding
        set_hw_encoding(False)

    if args.no_transition:
        args.transition_style = "none"
        args.sfx_kind = "none"

    if not os.path.isfile(args.video):
        print(f"ERROR: 영상 파일을 찾을 수 없습니다: {args.video}")
        sys.exit(1)

    range_text = args.ranges
    if args.ranges_file:
        with open(args.ranges_file, "r", encoding="utf-8") as f:
            range_text = f.read()

    try:
        labeled = parse_labeled_ranges(range_text)
    except ValueError as e:
        print(f"ERROR: 시간대 파싱 실패 - {e}")
        sys.exit(1)

    if not labeled:
        print("ERROR: 하이라이트 시간대를 하나 이상 입력하세요.")
        sys.exit(1)

    # 영상 길이를 벗어나는 구간은 잘라 맞춘다 (소제목은 유지).
    try:
        dur = get_duration(args.video)
        clipped = []
        for s, e, label in labeled:
            s = max(0.0, s)
            e = min(dur, e)
            if e - s >= 0.2:
                clipped.append((s, e, label))
            else:
                print(f"  (범위를 벗어나 건너뜀: {s:.1f}s ~ {e:.1f}s / 영상 {dur:.1f}s)")
        labeled = clipped
    except Exception as e:
        print(f"  (영상 길이 확인 실패, 입력값 그대로 사용: {e})")

    if not labeled:
        print("ERROR: 유효한 하이라이트 구간이 없습니다.")
        sys.exit(1)

    segments = [(s, e) for s, e, _ in labeled]

    # 소제목을 출력 타임라인(이어붙인 뒤 기준)으로 변환. 화면전환은 길이를
    # 바꾸지 않으므로 각 구간 길이를 누적하면 출력상의 표시 구간이 된다.
    label_windows = []
    _t = 0.0
    for s, e, label in labeled:
        d = e - s
        if label.strip():
            label_windows.append((_t, _t + d, label))
        _t += d

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = args.name.strip() or os.path.splitext(os.path.basename(args.video))[0]
    safe = safe_filename(base)
    out_video = str(output_dir / f"{safe}_highlight.mp4")
    out_srt = str(output_dir / f"{safe}_highlight.srt")
    out_chapters = str(output_dir / f"{safe}_chapters.txt")

    total = sum(e - s for s, e in segments)
    v_name = TRANSITION_STYLES.get(args.transition_style, args.transition_style)
    s_name = SFX_SPECS.get(args.sfx_kind, (None, None, 0, args.sfx_kind))[3]
    print(f"[1/{'3' if args.subtitles else '2'}] "
          f"{len(segments)}개 구간 컷 & 이어붙이기 "
          f"(총 {total:.1f}s / {total/60:.1f}분, 화면전환: {v_name} / 효과음: {s_name})")

    use_overlay = bool(label_windows)

    # Jump-cut changes timing, so label_windows (overlay) timings would be incorrect.
    # Disable jump-cut if labels are present.
    if args.jump_cut and use_overlay:
        print(f"  (소제목이 있으면 점프컷 비활성화 - 오버레이 타이밍 보존)")
        args.jump_cut = False

    with tempfile.TemporaryDirectory(prefix="manual_hl_") as tmpdir:
        # 소제목이 있으면 먼저 원본 컷을 임시로 만들고 오버레이를 입힌다.
        base_video = os.path.join(tmpdir, "highlight_raw.mp4") if use_overlay else out_video

        # compute_punchin_times if needed + merge with manual times
        from summarizer import parse_punchin_times, map_punchin_times

        punchins = {}
        manual_punchins = {}

        # 자동 펀치인
        if args.punchin != "none":
            punchins = compute_punchin_times(args.video, segments, args.punchin, tmpdir,
                                             closeup_sec=args.closeup_sec)

        # 수동 펀치인 파싱 및 매핑
        if args.punchin_times.strip():
            manual_times = parse_punchin_times(args.punchin_times)
            manual_punchins = map_punchin_times(manual_times, segments)

        # 병합: 수동 시간과 자동 시간이 3초 이내로 겹치면 자동 제거
        if manual_punchins and punchins:
            for seg_idx in manual_punchins:
                if seg_idx in punchins:
                    auto_times = punchins[seg_idx]
                    manual_times = manual_punchins[seg_idx]
                    filtered_auto = []
                    for at in auto_times:
                        keep = True
                        for mt in manual_times:
                            if abs(at - mt) <= 3.0:
                                keep = False
                                break
                        if keep:
                            filtered_auto.append(at)
                    punchins[seg_idx] = filtered_auto
                    if not filtered_auto:
                        del punchins[seg_idx]

        # 수동 + 자동 통합
        for seg_idx, times in manual_punchins.items():
            if seg_idx not in punchins:
                punchins[seg_idx] = times
            else:
                punchins[seg_idx].extend(times)
                punchins[seg_idx].sort()

        # 수동 펀치인 로그
        if manual_punchins:
            total_manual = sum(len(times) for times in manual_punchins.values())
            print(f"  펀치인(수동) {total_manual}곳: " +
                  ", ".join(f"{t:.1f}s" for times in manual_punchins.values() for t in sorted(times)))

        cut_and_concat(args.video, segments, base_video, tmpdir,
                       transition_style=args.transition_style,
                       sfx_kind=args.sfx_kind,
                       cam_region=args.cam_region,
                       closeup_sec=args.closeup_sec,
                       closeup_every=args.closeup_every,
                       punchins=punchins)

        # Apply jump-cut if requested
        if args.jump_cut:
            base_video_cut = os.path.join(tmpdir, "highlight_cut.mp4")
            cut_happened = silence_cut(base_video, base_video_cut, tmpdir)
            if cut_happened:
                base_video = base_video_cut

        if args.subtitles:
            print(f"[2/3] 완성 영상에서 오디오 추출...")
            wav_path = os.path.join(tmpdir, "audio.wav")
            extract_audio(base_video, wav_path)
            print(f"[3/3] Whisper 자막 생성 ({args.model})...")
            whisper_result = transcribe(wav_path, args.model, args.lang, args.prompt)
            out_dur = get_duration(base_video)
            srt_content = build_srt(whisper_result, [(0.0, out_dur)])
            with open(out_srt, "w", encoding="utf-8") as f:
                f.write(srt_content)

        if use_overlay:
            print(f"  소제목 {len(label_windows)}개 삽입 "
                  f"({WM_POSITIONS.get(args.label_pos, args.label_pos)})...")
            apply_overlays(base_video, out_video, tmpdir,
                           wm_pos=args.label_pos,
                           labels=label_windows, font=args.font,
                           label_size=args.label_size)

    # 유튜브 챕터 텍스트: 설명란에 그대로 붙여넣으면 챕터가 생긴다.
    with open(out_chapters, "w", encoding="utf-8") as f:
        f.write(build_chapters(segments))

    print(f"\nDone!")
    print(f"  Video    : {out_video}")
    if args.subtitles:
        print(f"  SRT      : {out_srt}")
    print(f"  Chapters : {out_chapters}")
    print(f"  (챕터 파일 내용을 유튜브 설명란에 붙여넣으면 구간 이동 챕터가 생깁니다)")


if __name__ == "__main__":
    main()

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
    apply_overlays,
    extract_audio,
    transcribe,
    build_srt,
    get_duration,
    safe_filename,
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
                        help="화면 전환: none / black / white (기본 black)")
    parser.add_argument("--sfx", dest="sfx_kind", default="whoosh",
                        choices=list(SFX_SPECS.keys()),
                        help="전환 효과음 (기본 whoosh)")
    parser.add_argument("--no-transition", action="store_true",
                        help="화면 전환/효과음 모두 끄기")
    parser.add_argument("--subtitles", action="store_true",
                        help="완성 영상에서 자막(SRT) 자동 생성 (Whisper)")
    parser.add_argument("--model", default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper 모델 (자막 켤 때만 사용, 기본 small)")
    parser.add_argument("--lang", default="ko", help="자막 언어 코드 (기본 ko)")
    parser.add_argument("--prompt", default=GAME_PROMPT,
                        help="Whisper initial_prompt (전문 용어 힌트)")
    parser.add_argument("--watermark", default="",
                        help="본영상에 새겨넣을 마크(로고/채널명) 이미지 경로")
    parser.add_argument("--wm-pos", default="tr", choices=list(WM_POSITIONS.keys()),
                        help="마크 위치: tl(좌상) tr(우상) bl(좌하) br(우하). 기본 tr")
    parser.add_argument("--wm-scale", type=float, default=0.12,
                        help="마크 가로폭 = 영상 가로폭 * 이 값 (기본 0.12)")
    parser.add_argument("--wm-margin", type=int, default=24,
                        help="마크 가장자리 여백(픽셀). 기본 24")
    parser.add_argument("--label-size", type=int, default=40,
                        help="하이라이트 소제목 글자 크기 (기본 40)")
    parser.add_argument("--font", default="Malgun Gothic",
                        help="소제목 글꼴 (기본 Malgun Gothic)")
    args = parser.parse_args()

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

    use_wm = bool(args.watermark) and os.path.isfile(args.watermark)
    use_overlay = use_wm or bool(label_windows)

    with tempfile.TemporaryDirectory(prefix="manual_hl_") as tmpdir:
        # 마크/소제목이 있으면 먼저 원본 컷을 임시로 만들고 오버레이를 입힌다.
        base_video = os.path.join(tmpdir, "highlight_raw.mp4") if use_overlay else out_video
        cut_and_concat(args.video, segments, base_video, tmpdir,
                       transition_style=args.transition_style,
                       sfx_kind=args.sfx_kind)

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
            wm_name = WM_POSITIONS.get(args.wm_pos, args.wm_pos)
            bits = []
            if use_wm:
                bits.append(f"마크 {wm_name}")
            if label_windows:
                bits.append(f"소제목 {len(label_windows)}개")
            print(f"  {' / '.join(bits)} 삽입...")
            apply_overlays(base_video, out_video, tmpdir,
                           watermark=args.watermark, wm_pos=args.wm_pos,
                           wm_scale=args.wm_scale, wm_margin=args.wm_margin,
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

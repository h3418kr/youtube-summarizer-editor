"""쇼츠(9:16 세로 영상) 내보내기 / Shorts (vertical 9:16) exporter.

로컬 영상(또는 요약본)에서 지정한 구간을 잘라 세로 1080x1920 쇼츠용
영상으로 변환한다. 유튜브 Shorts / Instagram Reels / TikTok 규격.

- 세로 변환 방식:
    center : 화면 중앙을 9:16 으로 크롭 (게임 화면 등 중앙에 시선이 있을 때)
    blur   : 원본을 그대로 두고 위아래를 블러 배경으로 채움 (화면 전체가 중요할 때)
- 여러 구간을 주면 하드컷으로 이어붙인다 (쇼츠 특성상 전환 효과 없음).
- --subtitles 를 켜면 완성본에서 Whisper 로 자막을 뽑아 크게 새겨넣는다.

시간대 입력 형식은 manual_highlight.py 와 동일 (한 줄에 'start - end').
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# 포터블(임베디드) 파이썬은 python311._pth 때문에 스크립트 폴더를 sys.path 에
# 자동 추가하지 않는다. 같은 폴더의 summarizer 등을 import 하려면 직접 넣어준다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from summarizer import (
    GAME_PROMPT,
    cut_and_concat,
    extract_audio,
    transcribe,
    build_srt,
    get_duration,
    safe_filename,
    copy_fonts_to,
)
from finalize import run_ffmpeg
from manual_highlight import parse_ranges

# 쇼츠 규격 (유튜브 Shorts / Reels / TikTok 공통)
SHORTS_W, SHORTS_H = 1080, 1920
SHORTS_MAX_SEC = 180  # 유튜브 Shorts 최대 3분

MODE_NAMES = {
    "smart": "스마트 자동",
    "center": "중앙 크롭",
    "left": "왼쪽 크롭",
    "right": "오른쪽 크롭",
    "blur": "블러 배경",
}

# 자막 세로 위치: key -> (사람이 읽는 이름, ASS Alignment, 기준 여백 MarginV)
#   하단은 아래에서, 상단은 위에서 그만큼 띄우고, 중앙은 화면 정중앙(여백 무시).
SUB_POS = {
    "bottom": ("하단", 2, 260),
    "center": ("중앙", 5, 0),
    "top": ("상단", 8, 260),
}


def _probe_video(src: str) -> tuple:
    """ffprobe를 써서 영상 너비/높이를 구한다. (iw, ih) 반환.

    Returns:
        (width, height) 또는 실패 시 (0, 0)
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", os.path.abspath(src)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return (0, 0)
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                w = stream.get("width", 0)
                h = stream.get("height", 0)
                if w > 0 and h > 0:
                    return (w, h)
    except Exception as e:
        print(f"  [smart-crop] ffprobe 실패: {e}")
    return (0, 0)


def _analyze_motion(src: str, iw: int, ih: int) -> int:
    """동작 분석으로 최적 수평 위치(crop_x) 찾기.

    Fallback: 실패 시 중앙 위치 반환.

    Args:
        src: 입력 영상 경로
        iw: 영상 너비
        ih: 영상 높이

    Returns:
        crop_x: 크롭 시작 x 좌표
    """
    cw = round(ih * SHORTS_W / SHORTS_H)  # 크롭 너비
    center_x = (iw - cw) // 2

    if iw <= 0 or ih <= 0 or cw <= 0:
        print(f"  [smart-crop] 영상 크기 오류: iw={iw}, ih={ih}, cw={cw}")
        return center_x

    # 프레임 추출 설정
    dw = 256  # 다운스케일된 너비
    dh = max(1, round(dw * ih / iw))

    try:
        # ffmpeg로 3fps 다운스케일 그레이스케일 프레임 추출
        cmd = [
            "ffmpeg", "-i", os.path.abspath(src),
            "-vf", f"fps=3,scale={dw}:-2,format=gray",
            "-f", "rawvideo", "-"
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30
        )

        frame_bytes = result.stdout
        frame_size = dw * dh
        n_frames = len(frame_bytes) // frame_size

        if n_frames < 2:
            print(f"  [smart-crop] 프레임 부족 (추출됨: {n_frames}개)")
            return center_x

        # 바이트를 numpy 배열로 reshape
        frames = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(n_frames, dh, dw)

        # 연속 프레임 간 동작 계산
        col_motion = np.zeros(dw, dtype=np.float32)
        for i in range(n_frames - 1):
            diff = np.abs(frames[i + 1].astype(np.float32) - frames[i].astype(np.float32))
            col_sum = np.sum(diff, axis=0)  # 각 열의 합
            col_motion += col_sum

        # 선택: 명암도(salience) 추가 — 각 프레임의 열별 표준편차 누적
        for i in range(n_frames):
            col_std = np.std(frames[i], axis=0)
            col_motion += col_std * 0.5  # 가중치 0.5

        # 이동 평균으로 부드럽게
        window_size = 5
        if dw > window_size:
            kernel = np.ones(window_size) / window_size
            col_motion = np.convolve(col_motion, kernel, mode='same')

        # 최적 크롭 윈도우 찾기 (누적합으로 O(n) 속도)
        cw_ds = max(1, round(cw * dw / iw))
        if cw_ds > dw:
            print(f"  [smart-crop] 크롭 너비가 프레임보다 큼: cw_ds={cw_ds}, dw={dw}")
            return center_x

        # 누적합 (cumsum)으로 윈도우 합 계산
        cumsum = np.concatenate(([0], np.cumsum(col_motion)))
        window_sums = cumsum[cw_ds:] - cumsum[:-cw_ds]

        # 최대 동작 구간 찾기
        best_i = np.argmax(window_sums)

        # 다운스케일 좌표를 원본으로 매핑
        crop_x = round(best_i * iw / dw)
        crop_x = max(0, min(crop_x, iw - cw))

        print(f"  [smart-crop] 최적 x={crop_x} (프레임={n_frames}, 최고 동작={window_sums[best_i]:.0f})")
        return crop_x

    except subprocess.TimeoutExpired:
        print(f"  [smart-crop] ffmpeg 타임아웃")
        return center_x
    except Exception as e:
        print(f"  [smart-crop] 분석 실패: {e}")
        return center_x


def _ass_time(t: str) -> str:
    """'HH:MM:SS,mmm' (SRT) -> 'H:MM:SS.cs' (ASS)."""
    hh, mm, rest = t.strip().split(":")
    ss, ms = rest.split(",")
    return f"{int(hh)}:{mm}:{ss}.{int(ms)//10:02d}"


def build_caption_ass(srt_content: str, w: int, h: int, font: str,
                      font_size: int, sub_pos: str) -> str:
    """Whisper SRT 를 세로 쇼츠용 ASS 자막으로 변환.

    PlayResX/Y 를 실제 영상 크기(1080x1920)로 지정하므로 FontSize·MarginV 가
    '실제 픽셀' 단위가 된다. (SRT 를 subtitles 필터에 바로 넣으면 기본 스크립트
    해상도 288 기준으로 렌더되어 글자가 6~7배로 확대되는 문제를 피한다.)
    """
    _name, align, margin_v = SUB_POS.get(sub_pos, SUB_POS["bottom"])
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\n"
        "ScaledBorderAndShadow: yes\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{font},{font_size},&H00FFFFFF,&HE0000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,4,2,{align},70,70,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )
    events = []
    for block in srt_content.strip().split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2 or "-->" not in lines[1]:
            continue
        start_s, end_s = [x.strip() for x in lines[1].split("-->")]
        text = "\\N".join(lines[2:]).replace("{", "(").replace("}", ")")
        events.append(
            f"Dialogue: 0,{_ass_time(start_s)},{_ass_time(end_s)},Cap,,0,0,0,,{text}")
    return head + "\n".join(events) + "\n"


def to_vertical(src: str, out_path: str, mode: str, ass_name: str = "",
                crop_x: int = None, cwd: str = None) -> None:
    """가로 영상을 1080x1920 세로로 변환하고, ass_name 이 있으면 자막도 새긴다.

    Args:
        src: 입력 영상
        out_path: 출력 경로
        mode: "blur" 또는 크롭 모드 (center, left, right, smart 등)
        ass_name: 자막 파일명 (선택)
        crop_x: 크롭 시작 x 좌표. None 이면 blur 모드만 사용 가능
        cwd: ffmpeg 작업 폴더 (자막 디렉토리)

    subtitles 필터 경로는 Windows 이스케이프가 까다로워 finalize 와 같은 방식으로
    자막 파일을 작업 폴더(cwd)에 두고 상대 경로로 참조한다.
    """
    sub_filter = f",subtitles={ass_name}:fontsdir=." if ass_name else ""

    if mode == "blur":
        # 배경: 화면을 꽉 채운 뒤 블러 / 전경: 원본 비율 그대로 가운데 배치
        fc = (f"[0:v]split[a][b];"
              f"[a]scale={SHORTS_W}:{SHORTS_H}:force_original_aspect_ratio=increase,"
              f"crop={SHORTS_W}:{SHORTS_H},boxblur=20:5[bg];"
              f"[b]scale={SHORTS_W}:-2[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1{sub_filter},format=yuv420p[v]")
    else:
        # 크롭 모드 (left, center, right, smart)
        if crop_x is not None:
            fc = (f"[0:v]crop='min(iw,ih*{SHORTS_W}/{SHORTS_H})':ih:{crop_x}:0,"
                  f"scale={SHORTS_W}:{SHORTS_H},setsar=1{sub_filter},format=yuv420p[v]")
        else:
            # 크롭 x가 없으면 중앙 기본값
            fc = (f"[0:v]crop='min(iw,ih*{SHORTS_W}/{SHORTS_H})':ih,"
                  f"scale={SHORTS_W}:{SHORTS_H},setsar=1{sub_filter},format=yuv420p[v]")

    cmd = ["ffmpeg", "-y", "-i", os.path.abspath(src),
           "-filter_complex", fc,
           "-map", "[v]", "-map", "0:a?",
           "-c:v", "libx264", "-preset", "fast", "-crf", "22",
           "-r", "30", "-fps_mode", "cfr",
           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
           "-movflags", "+faststart", os.path.abspath(out_path)]
    run_ffmpeg(cmd, label="(세로 변환)", cwd=cwd)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="로컬 영상 + 구간 -> 쇼츠(9:16 세로) 영상")
    parser.add_argument("video", help="로컬 영상 파일 경로")
    parser.add_argument("--ranges", default="",
                        help="쇼츠로 만들 구간(여러 줄). 각 줄 'start - end'")
    parser.add_argument("--ranges-file", default="",
                        help="구간 목록을 담은 텍스트 파일 경로")
    parser.add_argument("--output-dir", default="output", help="출력 폴더 (기본: output)")
    parser.add_argument("--name", default="",
                        help="출력 파일 이름(확장자 제외). 미지정 시 원본 파일명 사용")
    parser.add_argument("--mode", default="center",
                        choices=["smart", "center", "left", "right", "blur"],
                        help="세로 변환 방식: smart(자동) / center(중앙) / left(왼쪽) / "
                             "right(오른쪽) / blur(블러). 기본 center")
    parser.add_argument("--subtitles", action="store_true",
                        help="완성 쇼츠에서 자막(SRT) 자동 생성 후 크게 새겨넣기")
    parser.add_argument("--model", default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper 모델 (자막 켤 때만 사용, 기본 small)")
    parser.add_argument("--lang", default="ko", help="자막 언어 코드 (기본 ko)")
    parser.add_argument("--prompt", default=GAME_PROMPT,
                        help="Whisper initial_prompt (전문 용어 힌트)")
    parser.add_argument("--font", default="Paperlogy", help="자막 글꼴")
    parser.add_argument("--font-size", type=int, default=54,
                        help="자막 크기 (1080x1920 실제 픽셀 기준, 기본 54)")
    parser.add_argument("--sub-pos", default="bottom", choices=list(SUB_POS.keys()),
                        help="자막 세로 위치: bottom(하단) / center(중앙) / top(상단). 기본 하단")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"ERROR: 영상 파일을 찾을 수 없습니다: {args.video}")
        sys.exit(1)

    range_text = args.ranges
    if args.ranges_file:
        with open(args.ranges_file, "r", encoding="utf-8") as f:
            range_text = f.read()

    try:
        segments = parse_ranges(range_text)
    except ValueError as e:
        print(f"ERROR: 시간대 파싱 실패 - {e}")
        sys.exit(1)

    if not segments:
        print("ERROR: 쇼츠로 만들 구간을 하나 이상 입력하세요.")
        sys.exit(1)

    # 영상 길이를 벗어나는 구간은 잘라 맞춘다.
    try:
        dur = get_duration(args.video)
        clipped = []
        for s, e in segments:
            s = max(0.0, s)
            e = min(dur, e)
            if e - s >= 0.2:
                clipped.append((s, e))
            else:
                print(f"  (범위를 벗어나 건너뜀: {s:.1f}s ~ {e:.1f}s / 영상 {dur:.1f}s)")
        segments = clipped
    except Exception as e:
        print(f"  (영상 길이 확인 실패, 입력값 그대로 사용: {e})")

    if not segments:
        print("ERROR: 유효한 구간이 없습니다.")
        sys.exit(1)

    total = sum(e - s for s, e in segments)
    if total > SHORTS_MAX_SEC:
        print(f"  [주의] 총 길이 {total:.0f}초 - 유튜브 Shorts 최대는 {SHORTS_MAX_SEC}초(3분)입니다. "
              f"그대로 만들지만 Shorts 로는 올라가지 않을 수 있어요.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = args.name.strip() or os.path.splitext(os.path.basename(args.video))[0]
    safe = safe_filename(base)
    out_video = str(output_dir / f"{safe}_shorts.mp4")
    out_srt = str(output_dir / f"{safe}_shorts.srt")

    steps = 3 if args.subtitles else 2
    print(f"[1/{steps}] {len(segments)}개 구간 컷 & 이어붙이기 "
          f"(총 {total:.1f}s, 변환 방식: {MODE_NAMES[args.mode]})")

    with tempfile.TemporaryDirectory(prefix="shorts_") as tmpdir:
        # 1) 구간을 하드컷으로 이어붙인 가로 클립 (쇼츠는 전환 효과 없이 컷 편집이 기본)
        flat = os.path.join(tmpdir, "flat.mp4")
        cut_and_concat(args.video, segments, flat, tmpdir,
                       transition_style="none", sfx_kind="none")

        # 2) (선택) 자막 생성 — 이어붙인 짧은 클립에서 전사하므로 빠르다
        ass_name = ""
        if args.subtitles:
            print(f"[2/3] Whisper 자막 생성 ({args.model})...")
            wav_path = os.path.join(tmpdir, "audio.wav")
            extract_audio(flat, wav_path)
            whisper_result = transcribe(wav_path, args.model, args.lang, args.prompt)
            flat_dur = get_duration(flat)
            srt_content = build_srt(whisper_result, [(0.0, flat_dur)])
            if srt_content:
                # 사용자 제공용 SRT 는 그대로 저장
                with open(out_srt, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                # 번인용은 해상도/크기/위치를 정확히 제어하는 ASS 로 만든다
                ass_name = "captions.ass"
                pos_name = SUB_POS.get(args.sub_pos, SUB_POS["bottom"])[0]
                print(f"  자막 위치: {pos_name} / 크기: {args.font_size}px")
                ass = build_caption_ass(srt_content, SHORTS_W, SHORTS_H,
                                        args.font, args.font_size, args.sub_pos)
                with open(os.path.join(tmpdir, ass_name), "w", encoding="utf-8") as f:
                    f.write(ass)
            else:
                print("  (인식된 자막이 없어 자막 없이 진행)")

        # 3) 세로 변환 + 자막 번인
        print(f"[{steps}/{steps}] 1080x1920 세로 변환...")

        # 크롭 위치 결정
        crop_x = None
        if args.mode != "blur":
            iw, ih = _probe_video(flat)
            if iw > 0 and ih > 0:
                cw = round(ih * SHORTS_W / SHORTS_H)

                if args.mode == "smart":
                    crop_x = _analyze_motion(flat, iw, ih)
                elif args.mode == "left":
                    crop_x = 0
                elif args.mode == "center":
                    crop_x = (iw - cw) // 2
                elif args.mode == "right":
                    crop_x = iw - cw
                print(f"  크롭: {MODE_NAMES[args.mode]} (x={crop_x})")
            else:
                print(f"  (영상 크기 조회 실패, 중앙 크롭으로 진행)")
                crop_x = None

        # Copy bundled fonts to tmpdir so libass can find them
        if ass_name:
            copy_fonts_to(tmpdir)
        to_vertical(flat, out_video, args.mode, ass_name=ass_name, crop_x=crop_x, cwd=tmpdir)

    print(f"\nDone!")
    print(f"  Shorts : {out_video}")
    if args.subtitles and os.path.isfile(out_srt):
        print(f"  SRT    : {out_srt}")


if __name__ == "__main__":
    main()

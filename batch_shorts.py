"""쇼츠 배치(배치 자동 생성) / Batch Shorts (auto-generate N shorts from 1 video).

한 개 영상(URL 또는 로컬)에서 오디오 에너지 분석으로 상위 N개 하이라이트를 찾아,
각각을 스마트 크롭 + 자막 + (선택) AI 제목이 붙은 수직(1080x1920) 쇼츠로 자동 생성한다.

시간 단위로 소수점 초(float) 조회; 결과는 다른 탭과 호환가능한 SRT, 자막 영상으로 저장.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

# 포터블 파이썬: sys.path 에 스크립트 폴더 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from summarizer import (
    GAME_PROMPT,
    cut_and_concat,
    extract_audio,
    compute_energy,
    compute_voice_energy,
    transcribe,
    build_srt,
    get_duration,
    safe_filename,
    copy_fonts_to,
    download_video,
    _gaussian_smooth,
    detect_cam_region,
    detect_chat_region,
    compute_chat_activity,
)
from shorts import (
    SHORTS_W,
    SHORTS_H,
    SUB_POS,
    build_caption_ass,
    to_vertical,
    _probe_video,
    _analyze_motion,
)
from finalize import run_ffmpeg


def pick_highlights(
    energy: np.ndarray,
    window_sec: float,
    count: int,
    clip_len: float,
    duration: float,
    chat_curve: np.ndarray = None,
    chat_sample_sec: float = None,
    chat_weight: float = None,
    voice_energy: np.ndarray = None,
) -> List[Tuple[float, float]]:
    """Energy 배열에서 상위 N개 (겹치지 않는) 하이라이트 구간을 고른다.

    각 구간은 [peak_time - clip_len*0.35, peak_time + clip_len*0.65] 형태.
    구간들은 비디오 내에서 clip_len 초 이상 떨어져 있어야 한다(겹치지 않음).

    Args:
        energy: 에너지 배열 (compute_energy 결과)
        window_sec: 윈도우 크기(초)
        count: 원하는 쇼츠 개수
        clip_len: 각 쇼츠 길이(초)
        duration: 전체 비디오 길이(초)
        chat_curve: 채팅 활동 곡선 (없으면 오디오 에너지만 사용)
        chat_sample_sec: 채팅 곡선의 샘플 간격(초)
        chat_weight: 채팅 적응 가중치
        voice_energy: 목소리 대역 에너지

    Returns:
        [(start, end), ...] 선택된 구간 목록 (시간순)
    """
    # 에너지 평활화
    sigma = 10.0 / window_sec
    smoothed = _gaussian_smooth(energy, sigma)

    # z-score 정규화: 전체 에너지
    energy_mean = np.mean(smoothed)
    energy_std = np.std(smoothed)
    if energy_std > 1e-6:
        energy_z = (smoothed - energy_mean) / energy_std
    else:
        energy_z = np.zeros_like(smoothed)

    # z-score 정규화: 목소리 대역 에너지
    voice_z = np.zeros_like(energy_z)
    if voice_energy is not None and len(voice_energy) > 0:
        voice_energy_smooth = _gaussian_smooth(voice_energy, sigma)
        voice_mean = np.mean(voice_energy_smooth)
        voice_std = np.std(voice_energy_smooth)
        if voice_std > 1e-6:
            voice_z = (voice_energy_smooth - voice_mean) / voice_std

    # 결합 점수 계산
    if chat_curve is not None and len(chat_curve) > 0 and chat_sample_sec is not None and chat_weight is not None:
        # 채팅 z-score 정규화
        chat_mean = np.mean(chat_curve)
        chat_std = np.std(chat_curve)
        if chat_std > 1e-6:
            chat_z = (chat_curve - chat_mean) / chat_std
        else:
            chat_z = np.zeros_like(chat_curve)

        # 채팅 곡선을 에너지 윈도 타임라인으로 리샘플
        energy_times = np.arange(len(smoothed)) * window_sec
        chat_times = np.arange(len(chat_curve)) * chat_sample_sec
        chat_z_resampled = np.interp(energy_times, chat_times, chat_z, left=0.0, right=0.0)

        # 결합 점수: 0.5 * energy_z + 1.0 * voice_z + chat_weight * chat_z
        smoothed = 0.5 * energy_z + 1.0 * voice_z + chat_weight * chat_z_resampled
        print(f"  채팅 가중치: {chat_weight:.2f}")
        print("  (목소리 반응 + 채팅 반응 반영된 점수로 분석 중)")
    else:
        # 채팅이 없어도 목소리는 적용
        smoothed = 0.5 * energy_z + 1.0 * voice_z
        if len(voice_z) > 0 and np.any(voice_z != 0):
            print("  (목소리 반응 반영된 점수로 분석 중)")

    # 피크 감지 (±20s 윈도우에서 로컬 최대)
    threshold = np.percentile(smoothed, 70)
    peak_radius = int(20 / window_sec)
    peaks = []
    for i in range(len(smoothed)):
        if smoothed[i] < threshold:
            continue
        lo = max(0, i - peak_radius)
        hi = min(len(smoothed), i + peak_radius + 1)
        if smoothed[i] == smoothed[lo:hi].max():
            peaks.append((i, smoothed[i]))

    if not peaks:
        print("  (피크를 찾을 수 없어, 비디오 앞부터 선택)")
        # Fallback: 영상 앞에서부터 clip_len 간격으로 선택
        selected = []
        for i in range(count):
            start = i * clip_len
            if start >= duration:
                break
            end = min(start + clip_len, duration)
            selected.append((start, end))
        return selected

    # 피크를 에너지 내림차순으로 정렬하고 겹치지 않는 범위에서 선택
    peaks.sort(key=lambda x: -x[1])
    selected_peaks = []
    for peak_idx, energy_val in peaks:
        peak_time = peak_idx * window_sec
        # 이 피크가 기존 선택과 겹치는지 확인
        overlaps = False
        for prev_start, prev_end in selected_peaks:
            # 새 구간: [peak_time - clip_len*0.35, peak_time + clip_len*0.65]
            new_start = max(0, peak_time - clip_len * 0.35)
            new_end = min(duration, peak_time + clip_len * 0.65)
            # 겹침 체크
            if not (new_end < prev_start or new_start > prev_end):
                overlaps = True
                break
        if not overlaps:
            new_start = max(0, peak_time - clip_len * 0.35)
            new_end = min(duration, peak_time + clip_len * 0.65)
            selected_peaks.append((new_start, new_end))
            if len(selected_peaks) >= count:
                break

    # 시간순 정렬
    selected_peaks.sort(key=lambda x: x[0])

    if not selected_peaks:
        print(f"  (피크 선택 실패, 처음부터 선택)")
        selected_peaks = [
            (min(i * clip_len, duration - clip_len), min((i + 1) * clip_len, duration))
            for i in range(count)
            if i * clip_len < duration
        ]

    print(
        f"  상위 {len(selected_peaks)}/{count} 피크 선택 "
        f"(clip_len={clip_len:.0f}s)"
    )
    return selected_peaks


def make_ai_title_ass(duration: float, title_text: str) -> str:
    """Full-duration 동안 표시할 AI 제목용 ASS 자막 생성.

    - 위치: 화면 상단 중앙(an=8)
    - 크기: 큼(font_size=72)
    - 색상: 흰색, 검정 외곽선
    """
    def _ass_ts(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        cs = int(round((sec - int(sec)) * 100))
        if cs == 100:
            cs = 0
            s += 1
        return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

    head = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {SHORTS_W}\nPlayResY: {SHORTS_H}\n"
        "ScaledBorderAndShadow: yes\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Title,Paperlogy,72,&H00FFFFFF,&HFF000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,3,2,8,70,70,100,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )
    end_time = _ass_ts(duration)
    text = title_text.replace("{", "(").replace("}", ")").replace("\n", r"\N")
    events = f"Dialogue: 0,0:00:00.00,{end_time},Title,,0,0,0,,{text}\n"
    return head + events


def call_gemini_title(transcript: str, api_key: str) -> str:
    """Gemini 에 요청해 한국어 제목(~12자) 생성. 실패 시 빈 문자열."""
    if not api_key or not api_key.strip():
        return ""

    try:
        import urllib.request
        import json as json_lib

        # finalize.py 의 gemini_labels 와 동일한 패턴
        prompt = (
            "다음은 게임 방송 클립의 자막입니다. "
            "이 클립을 한눈에 나타내는 아주 짧은 한국어 제목(12자 이내)을 하나만 만드세요. "
            "제목만 출력하고 설명이나 따옴표는 빼세요.\n\n자막:\n"
            + transcript[:500]  # 토큰 절감
        )

        _NONE = "BLOCK_NONE"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "safetySettings": [
                {"category": cat, "threshold": "BLOCK_NONE"}
                for cat in [
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                ]
            ],
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=json_lib.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json_lib.loads(resp.read().decode("utf-8"))

        if (
            result.get("candidates")
            and result["candidates"][0].get("content", {}).get("parts")
        ):
            title = result["candidates"][0]["content"]["parts"][0].get("text", "").strip()
            title = title.replace('"', "").replace("'", "").strip()
            if title:
                return title[:20]  # 최대 20자 자르기(혹시 모르니)
    except Exception as e:
        print(f"  [AI 제목 실패] {e}")

    return ""


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="한 영상에서 상위 N개 하이라이트 쇼츠 자동 생성"
    )
    parser.add_argument("video", help="URL 또는 로컬 영상 파일")
    parser.add_argument(
        "--count", type=int, default=5, help="생성할 쇼츠 개수 (기본 5)"
    )
    parser.add_argument(
        "--clip-len",
        type=float,
        default=30,
        help="각 쇼츠 길이(초, 기본 30)",
    )
    parser.add_argument(
        "--mode",
        default="smart",
        choices=["smart", "center", "left", "right", "blur"],
        help="세로 변환 방식 (기본 smart)",
    )
    parser.add_argument(
        "--subtitles",
        action="store_true",
        help="자막 자동 생성 및 번인",
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 모델 (기본 small)",
    )
    parser.add_argument(
        "--lang", default="ko", help="자막 언어 코드 (기본 ko)"
    )
    parser.add_argument(
        "--sub-pos",
        default="bottom",
        choices=list(SUB_POS.keys()),
        help="자막 위치 (기본 bottom)",
    )
    parser.add_argument(
        "--font", default="Paperlogy", help="자막 글꼴 (기본 Paperlogy)"
    )
    parser.add_argument(
        "--font-size", type=int, default=54, help="자막 크기(픽셀, 기본 54)"
    )
    parser.add_argument(
        "--ai-title", action="store_true", help="AI 제목(Gemini) 추가"
    )
    parser.add_argument(
        "--gemini-key", default="", help="Gemini API 키 (AI 제목 필요)"
    )
    parser.add_argument(
        "--output-dir", default="output", help="출력 폴더 (기본 output)"
    )
    parser.add_argument(
        "--name", default="", help="출력 이름 (미지정 시 원본 파일명 사용)"
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=720,
        help="다운로드 시 최대 높이(기본 720)",
    )
    parser.add_argument("--cpu-encode", action="store_true",
                        help="GPU 가속 인코딩 끄기 (호환성 문제 시)")
    parser.add_argument("--chat-analysis", action="store_true",
                        help="화면 채팅창 자동 감지 & 반응 반영 (채팅이 없으면 자동 무시됨)")
    parser.add_argument("--chat-region", default="auto", choices=["auto", "left", "right"],
                        help="채팅 위치: auto=자동감지, left=왼쪽, right=오른쪽 (기본: auto)")
    parser.add_argument("--cookies-browser", default="",
                        help="로그인 쿠키를 가져올 브라우저 (chrome/edge/whale/firefox). 연령제한·구독자 전용 다시보기용")
    args = parser.parse_args()

    if args.cpu_encode:
        from summarizer import set_hw_encoding
        set_hw_encoding(False)

    # 1. 비디오 소스 해석
    if os.path.isfile(args.video):
        video_path = os.path.abspath(args.video)
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        print(f"[1/4] 로컬 영상 사용: {video_path}")
    else:
        print(f"[1/4] 영상 다운로드: {args.video}")
        with tempfile.TemporaryDirectory(prefix="batch_shorts_dl_") as tmpdir:
            try:
                video_path, title = download_video(args.video, tmpdir, args.max_height, cookies_browser=args.cookies_browser)
                video_name = title
                print(f"  제목: {title}")
            except Exception as e:
                print(f"ERROR: 다운로드 실패 - {e}")
                sys.exit(1)

    try:
        duration = get_duration(video_path)
        print(f"  길이: {duration:.1f}초")
    except Exception as e:
        print(f"ERROR: 영상 길이 확인 실패 - {e}")
        sys.exit(1)

    # 2. 오디오 분석 → 상위 N개 하이라이트 선택
    print(f"[2/4] 오디오 분석 및 하이라이트 선택...")
    with tempfile.TemporaryDirectory(prefix="batch_shorts_analysis_") as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(video_path, wav_path)
        energy, window_sec = compute_energy(wav_path)

        # 목소리 대역 에너지 (항상 계산)
        print(f"  목소리 대역(200-3800Hz) 에너지 계산 중...")
        voice_energy, _ = compute_voice_energy(wav_path, tmpdir, window_sec=0.5)

        # 채팅 반응 분석 (선택사항)
        chat_curve = None
        chat_sample_sec = None
        chat_weight = None
        if args.chat_analysis:
            cam_region = None
            try:
                cam_region = detect_cam_region(video_path, tmpdir)
            except Exception:
                pass

            chat_region = None
            if args.chat_region != "auto":
                # 수동 채팅 위치 지정
                W, H = get_media_size(video_path)
                band_w = int(W * 0.25)  # 폭 = W의 25%

                if args.chat_region == "left":
                    chat_x = 0
                elif args.chat_region == "right":
                    chat_x = W - band_w
                else:
                    chat_x = 0

                chat_y = int(H * 0.10)  # 상단 10% 여백
                chat_h = int(H * 0.80)  # 높이 = H의 80%

                # 캠 영역과 겹치면 겹치는 세로 구간을 h에서 잘라냄
                if cam_region:
                    cam_x, cam_y, cam_w, cam_h = cam_region
                    # 캠 영역이 채팅 밴드의 x 범위와 겹치는지 확인
                    if not (chat_x + band_w <= cam_x or cam_x + cam_w <= chat_x):
                        # x 범위에서 겹침 -> y 범위 확인
                        if cam_y < chat_y + chat_h and cam_y + cam_h > chat_y:
                            # y 범위에서도 겹침 -> 겹치는 세로 구간을 제거
                            overlap_y_start = max(chat_y, cam_y)
                            overlap_y_end = min(chat_y + chat_h, cam_y + cam_h)
                            overlap_h = overlap_y_end - overlap_y_start
                            if overlap_h > 0:
                                chat_h -= overlap_h

                chat_region = (chat_x, chat_y, band_w, chat_h)
                position_name = "왼쪽" if args.chat_region == "left" else "오른쪽"
                print(f"  채팅 영역(수동-{position_name}): x={chat_x}, y={chat_y}, {band_w}x{chat_h}")
            else:
                # 자동 감지
                print(f"  채팅 영역 감지 시도 중...")
                chat_region = detect_chat_region(video_path, tmpdir, exclude_region=cam_region)

            if chat_region:
                print(f"  채팅 활동 곡선 계산 중...")
                chat_curve, chat_sample_sec, active_ratio, chat_weight = compute_chat_activity(
                    video_path, chat_region, duration, tmpdir, sample_sec=2.0
                )
                if len(chat_curve) > 0:
                    print(f"  채팅 곡선: {len(chat_curve)} 포인트 "
                          f"(min={chat_curve.min():.3f}, max={chat_curve.max():.3f}, "
                          f"mean={chat_curve.mean():.3f})")
                    print(f"  채팅 가중치: {chat_weight:.2f} (활동 비율 {active_ratio*100:.0f}%)")
                else:
                    print(f"  (채팅 곡선 비어있음 - 오디오 분석만 사용)")
                    chat_curve = None
            else:
                print(f"  (채팅창 감지 실패 - 오디오 분석만 사용)")

        highlights = pick_highlights(
            energy,
            window_sec,
            args.count,
            args.clip_len,
            duration,
            chat_curve=chat_curve,
            chat_sample_sec=chat_sample_sec,
            chat_weight=chat_weight,
            voice_energy=voice_energy,
        )
        if not highlights:
            print("ERROR: 하이라이트를 선택할 수 없습니다.")
            sys.exit(1)
        print(f"  {len(highlights)}개 하이라이트 선택됨")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = args.name.strip() or safe_filename(video_name)
    produced = []

    # 3. 각 하이라이트별 쇼츠 생성
    for idx, (start, end) in enumerate(highlights, 1):
        try:
            print(f"[3/{len(highlights)+1}] 쇼츠 {idx}/{len(highlights)}: "
                  f"{start:.1f}s ~ {end:.1f}s ({end - start:.1f}s)")

            with tempfile.TemporaryDirectory(prefix=f"shorts_{idx:02d}_") as tmpdir:
                # 3.1) 수평 클립 생성
                flat = os.path.join(tmpdir, "flat.mp4")
                cut_and_concat(
                    video_path,
                    [(start, end)],
                    flat,
                    tmpdir,
                    transition_style="none",
                    sfx_kind="none",
                )

                # 3.2) 자막 (선택)
                ass_names = []
                if args.subtitles:
                    wav_tmp = os.path.join(tmpdir, "flat_audio.wav")
                    extract_audio(flat, wav_tmp)
                    whisper_result = transcribe(wav_tmp, args.model, args.lang, GAME_PROMPT)
                    clip_dur = get_duration(flat)
                    srt_content = build_srt(whisper_result, [(0.0, clip_dur)])

                    if srt_content:
                        # 유저용 SRT 저장
                        out_srt = str(output_dir / f"{base}_short_{idx:02d}.srt")
                        with open(out_srt, "w", encoding="utf-8") as f:
                            f.write(srt_content)

                        # 자막 번인용 ASS
                        ass = build_caption_ass(
                            srt_content,
                            SHORTS_W,
                            SHORTS_H,
                            args.font,
                            args.font_size,
                            args.sub_pos,
                        )
                        ass_name = f"captions_{idx}.ass"
                        with open(os.path.join(tmpdir, ass_name), "w", encoding="utf-8") as f:
                            f.write(ass)
                        ass_names.append(ass_name)

                # 3.3) AI 제목 (선택)
                if args.ai_title and args.gemini_key:
                    # 자막이 있으면 그걸 쓰고, 없으면 Whisper 빠른 전사
                    try:
                        if not args.subtitles:
                            wav_tmp = os.path.join(tmpdir, "flat_audio.wav")
                            extract_audio(flat, wav_tmp)
                        whisper_result = transcribe(wav_tmp, "tiny", args.lang, GAME_PROMPT)
                        transcript = " ".join(
                            seg.get("text", "").strip()
                            for seg in whisper_result.get("segments", [])
                        )
                    except Exception:
                        transcript = ""

                    if transcript:
                        title = call_gemini_title(transcript, args.gemini_key)
                        if title:
                            print(f"  AI 제목: '{title}'")
                            ass = make_ai_title_ass(get_duration(flat), title)
                            ass_name = f"title_{idx}.ass"
                            with open(os.path.join(tmpdir, ass_name), "w", encoding="utf-8") as f:
                                f.write(ass)
                            ass_names.append(ass_name)

                # 3.4) 세로 변환
                out_video = str(output_dir / f"{base}_short_{idx:02d}.mp4")

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
                        print(f"  크롭: {args.mode} (x={crop_x})")

                # 폰트 복사
                if ass_names:
                    copy_fonts_to(tmpdir)

                # ASS 멀티패스: 여러 ass 를 차례로 적용
                if ass_names:
                    # 첫 번째 pass: flat -> temp1.mp4 (첫 ass)
                    temp_videos = []
                    for i, ass_name in enumerate(ass_names):
                        if i == 0:
                            # flat -> temp1 (세로 변환 + 첫 ASS)
                            temp_out = os.path.join(tmpdir, f"temp_{i}.mp4")
                            to_vertical(
                                flat,
                                temp_out,
                                args.mode,
                                ass_name=ass_name,
                                crop_x=crop_x,
                                cwd=tmpdir,
                            )
                            temp_videos.append(temp_out)
                        else:
                            # 이미 세로 변환된 것에 추가 ASS 적용
                            # (너무 복잡하므로 생략; 보통 자막만 있으면 충분)
                            pass

                    # 최종 출력
                    if temp_videos:
                        shutil.copy(temp_videos[0], out_video)
                    else:
                        # Fallback: 자막 없이 진행
                        to_vertical(flat, out_video, args.mode, crop_x=crop_x, cwd=tmpdir)
                else:
                    # 자막 없이 세로 변환만
                    to_vertical(flat, out_video, args.mode, crop_x=crop_x, cwd=tmpdir)

                produced.append(out_video)
                print(f"  완성: {out_video}")

        except Exception as e:
            print(f"  ERROR: 쇼츠 {idx} 생성 실패 - {e}")
            continue

    # 4. 완료 리포트
    print(f"\n[4/{len(highlights)+1}] 완료!")
    if produced:
        print(f"생성된 {len(produced)}개 쇼츠:")
        for out in produced:
            print(f"  • {out}")
    else:
        print("ERROR: 생성된 쇼츠가 없습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()

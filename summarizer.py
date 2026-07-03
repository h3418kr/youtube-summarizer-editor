"""YouTube video summarizer - downloads video, finds high-energy segments, creates ~10min summary with subtitles."""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _setup_bundled_paths():
    """포터블 배포용: 스크립트 폴더 옆의 ffmpeg/bin, python 패키지를 PATH에 추가."""
    base = os.path.dirname(os.path.abspath(__file__))
    for rel in (os.path.join("ffmpeg", "bin"), "ffmpeg"):
        p = os.path.join(base, rel)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "ffmpeg.exe")):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            break


_setup_bundled_paths()

import whisper
from pydub import AudioSegment


# Windows: 하위 프로세스가 별도 콘솔(검은 창)을 띄우지 않도록.
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# 모든 하위 프로세스 공통 옵션:
#   stdin=DEVNULL      : ffmpeg/yt-dlp 가 대화형 입력을 기다리며 멈추는 것을 방지
#   creationflags      : 콘솔 창 숨김
_PROC_KW = dict(stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW)


def run(cmd, **kwargs):
    kw = dict(_PROC_KW)
    kw.update(kwargs)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)
    return result.stdout.strip()


def run_ffmpeg(cmd, label: str = "") -> None:
    """ffmpeg 실행: 콘솔창 숨김 + stdin 차단 + 진행상황(time=) 스트리밍.

    긴 인코딩 중에도 '멈춘 것처럼' 보이지 않도록 마지막 진행 줄을 출력한다.
    """
    # ffmpeg 는 -nostdin 으로 표준입력을 아예 건드리지 않게 한다.
    if cmd and "ffmpeg" in os.path.basename(str(cmd[0])).lower():
        cmd = [cmd[0], "-nostdin", "-hide_banner"] + list(cmd[1:])

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        **_PROC_KW,
    )
    last = ""
    for raw in proc.stdout:
        line = raw.rstrip("\r\n")
        if not line:
            continue
        low = line.lower()
        if "time=" in line or "error" in low or "invalid" in low or "failed" in low:
            last = line
            print(f"    {label} {line}".rstrip(), flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=last)


def download_video(url: str, tmpdir: str, max_height: int = 720) -> Tuple[str, str]:
    info_raw = run(["yt-dlp", "--print", "%(id)s|||%(title)s", "--no-playlist", url])
    vid_id, title = info_raw.split("|||", 1)
    out_path = os.path.join(tmpdir, f"{vid_id}.mp4")
    fmt = (
        f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]"
        f"/bestvideo[height<={max_height}]+bestaudio"
        f"/best[height<={max_height}]/best"
    )
    subprocess.run(
        ["yt-dlp", "-f", fmt, "--merge-output-format", "mp4",
         "--newline",
         "-o", out_path, "--no-playlist", "--no-update", url],
        check=True, **_PROC_KW
    )
    return out_path, title


def extract_audio(video_path: str, out_wav: str) -> None:
    run_ffmpeg(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", "-f", "wav", out_wav],
        label="(오디오 추출)",
    )


def get_duration(path: str) -> float:
    out = run(["ffprobe", "-v", "quiet", "-print_format", "json",
               "-show_format", path])
    data = json.loads(out)
    return float(data["format"]["duration"])


def _bundled_model_root():
    """포터블 배포용: 스크립트 폴더 옆의 models 폴더가 있으면 사용."""
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base, "models")
    return p if os.path.isdir(p) else None


# Whisper 가 게임(디아블로2) 전문 용어를 잘 알아듣도록 유도하는 힌트 문장.
# initial_prompt 로 넣으면 룬/룬워드/아이템/클래스 이름을 문맥으로 인식해
# "참 눈→참 룬", "햇불→횃불", "레다→래더" 같은 오인식이 크게 줄어든다.
# (Whisper 프롬프트는 약 224토큰까지만 반영되므로 핵심 용어 위주로 짧게 유지)
GAME_PROMPT = (
    "디아블로2 래더 하드코어 방송입니다. 룬, 룬워드, 텔포, 소켓, 저항, 도박 이야기를 합니다. "
    "룬 이름: 엘 티르 랄 오르 솔 암 샤엘 헬 코 팔 펄 움 말 이스트 굴 벡스 옴 로 수르 베르 자 참 조드. "
    "룬워드: 인피니티 에니그마 스피릿 그리프 포티튜드 하트오브오크 콜투암즈 모자이크 스텔스 로어. "
    "아이템: 소저 샤코 마라 그리폰의눈 스톰실드 횃불 애니 조던의돌. "
    "클래스: 바바리안 팔라딘 소서리스 네크로맨서 아마존 드루이드 어쌔신."
)


def transcribe(wav_path: str, model_name: str, lang: str, prompt: str = None):
    print(f"  Transcribing with Whisper ({model_name})...")
    model = whisper.load_model(model_name, download_root=_bundled_model_root())
    result = model.transcribe(
        wav_path, language=lang, word_timestamps=True, verbose=False,
        initial_prompt=prompt or None,
    )
    return result


def compute_energy(wav_path: str, window_sec: float = 0.5) -> Tuple[np.ndarray, float]:
    audio = AudioSegment.from_wav(wav_path)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    window = int(sr * window_sec)
    n_windows = len(samples) // window
    energy = np.zeros(n_windows)
    for i in range(n_windows):
        chunk = samples[i * window:(i + 1) * window]
        energy[i] = np.sqrt(np.mean(chunk ** 2))
    return energy, window_sec


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    kernel_size = int(6 * sigma) | 1  # ensure odd
    half = kernel_size // 2
    k = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same")


def find_exciting_segments(
    energy: np.ndarray,
    window_sec: float,
    whisper_result,
    target_sec: int = 600,
    expand_before: float = 5.0,
    expand_after: float = 20.0,
    bridge_gap: float = 8.0,
) -> List[Tuple[float, float]]:
    """Find high-energy segments that sum to approximately target_sec.

    bridge_gap: 선택된 하이라이트끼리 원본상 시간차가 이 값(초) 이하이면
                같은 내용으로 보고 사이 구간까지 포함해 하나로 이어붙인다.
                (전환 효과는 이렇게 병합된 최종 구간들 '사이'에만 들어간다.)
    """
    sigma = 10.0 / window_sec  # smooth over ~10s
    smoothed = _gaussian_smooth(energy, sigma)

    threshold = np.percentile(smoothed, 60)

    # Local max peak detection in ±20s window
    peak_radius = int(20 / window_sec)
    peaks = []
    for i in range(len(smoothed)):
        if smoothed[i] < threshold:
            continue
        lo = max(0, i - peak_radius)
        hi = min(len(smoothed), i + peak_radius + 1)
        if smoothed[i] == smoothed[lo:hi].max():
            peaks.append(i)

    # Build Whisper boundary lookup
    seg_starts = []
    seg_ends = []
    for seg in whisper_result.get("segments", []):
        seg_starts.append(seg["start"])
        seg_ends.append(seg["end"])

    def snap_start(t: float) -> float:
        if not seg_starts:
            return t
        idx = min(range(len(seg_starts)), key=lambda i: abs(seg_starts[i] - t))
        return seg_starts[idx] if abs(seg_starts[idx] - t) < 3.0 else t

    def snap_end(t: float) -> float:
        if not seg_ends:
            return t
        idx = min(range(len(seg_ends)), key=lambda i: abs(seg_ends[i] - t))
        return seg_ends[idx] if abs(seg_ends[idx] - t) < 3.0 else t

    # Expand each peak into a segment
    raw_segments = []
    total_dur = len(energy) * window_sec
    for p in peaks:
        center = p * window_sec
        start = snap_start(max(0.0, center - expand_before))
        end = snap_end(min(total_dur, center + expand_after))
        raw_segments.append((start, end, float(smoothed[p])))

    # Merge overlapping
    raw_segments.sort(key=lambda x: x[0])
    merged = []
    for start, end, score in raw_segments:
        if merged and start <= merged[-1][1]:
            prev_start, prev_end, prev_score = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), max(prev_score, score))
        else:
            merged.append((start, end, score))

    # Greedy select by score until target_sec reached
    merged.sort(key=lambda x: -x[2])
    selected = []
    total = 0.0
    for start, end, score in merged:
        dur = end - start
        if total + dur > target_sec * 1.2:
            continue
        selected.append((start, end))
        total += dur
        if total >= target_sec:
            break

    selected.sort(key=lambda x: x[0])

    # 시간차가 짧은(같은 내용) 하이라이트끼리 사이 구간까지 포함해 하나로 병합
    bridged: List[List[float]] = []
    for s, e in selected:
        if bridged and s - bridged[-1][1] <= bridge_gap:
            bridged[-1][1] = max(bridged[-1][1], e)
        else:
            bridged.append([s, e])
    final = [(s, e) for s, e in bridged]

    total = sum(e - s for s, e in final)
    print(f"  Selected {len(selected)} peaks -> {len(final)} clips after bridging "
          f"(gap<={bridge_gap:.0f}s), totaling {total:.1f}s ({total/60:.1f} min)")
    return final


def _is_noise(text: str) -> bool:
    """Whisper 환각(hallucination) 및 노이즈 텍스트 감지."""
    if not text or len(text) < 2:
        return True
    # 한국어·일본어·영어·숫자·공백·기본 구두점 이외 문자가 많으면 노이즈
    clean = re.sub(r'[가-힣぀-ヿ一-鿿a-zA-Z0-9\s\.,!?~\-\'\"()]', '', text)
    if len(clean) > len(text) * 0.3:
        return True
    return False


def build_srt(whisper_result, segments: List[Tuple[float, float]],
              merge_gap: float = 0.8, min_dur: float = 1.2, max_chars: int = 45) -> str:
    """
    Build SRT with:
    - Timeline remapping to concatenated output
    - Short gap merging (merge_gap seconds)
    - Minimum display duration (min_dur seconds)
    - Noise / hallucination filtering
    """
    def fmt_time(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # Build output timeline mapping
    output_offsets = []
    cumulative = 0.0
    for start, end in segments:
        output_offsets.append((start, end, cumulative))
        cumulative += end - start

    # 1단계: Whisper 세그먼트를 출력 타임라인으로 매핑
    raw = []  # (out_start, out_end, text)
    for wseg in whisper_result.get("segments", []):
        ws = wseg["start"]
        we = wseg["end"]
        text = wseg["text"].strip()
        if not text or _is_noise(text):
            continue
        if we - ws < 0.15:  # 너무 짧은 세그먼트 제거
            continue
        for orig_start, orig_end, out_off in output_offsets:
            if ws >= orig_start and we <= orig_end:
                raw.append((out_off + (ws - orig_start), out_off + (we - orig_start), text))
                break

    if not raw:
        return ""

    # 2단계: 짧은 간격의 세그먼트 병합
    merged = [list(raw[0])]  # [start, end, text]
    for s, e, t in raw[1:]:
        prev = merged[-1]
        gap = s - prev[1]
        combined = prev[2] + " " + t
        if gap <= merge_gap and len(combined) <= max_chars:
            prev[1] = e
            prev[2] = combined
        else:
            # 이전 항목이 너무 길면 max_chars 기준으로 줄 바꿈
            merged.append([s, e, t])

    # 3단계: 최소 표시 시간 보장
    entries = []
    for i, (s, e, t) in enumerate(merged):
        dur = e - s
        if dur < min_dur:
            e = s + min_dur
        entries.append(f"{i+1}\n{fmt_time(s)} --> {fmt_time(e)}\n{t}\n")

    return "\n".join(entries)


# 화면 전환(비디오) 스타일: key -> 사람이 읽는 이름
TRANSITION_STYLES = {
    "none":  "없음",
    "black": "암전(fade to black)",
    "white": "화이트 플래시",
}

# 전환 효과음(오디오) 종류: key -> (lavfi 입력, 오디오 필터, 길이(초), 사람이 읽는 이름)
SFX_SPECS = {
    "none": (None, None, 0.0, "없음"),
    "whoosh": (
        "anoisesrc=d=0.5:c=pink:a=0.85:r=44100",
        "afade=t=in:st=0:d=0.25,afade=t=out:st=0.25:d=0.25,"
        "highpass=f=250,lowpass=f=5500,volume=1.6",
        0.5, "휙(whoosh)",
    ),
    "swoosh": (
        "anoisesrc=d=0.45:c=white:a=0.8:r=44100",
        "afade=t=in:st=0:d=0.30,afade=t=out:st=0.15:d=0.30,"
        "highpass=f=600,lowpass=f=9000,volume=1.4",
        0.45, "스와이프(swoosh)",
    ),
    "beep": (
        "sine=f=880:d=0.25:r=44100",
        "afade=t=in:st=0:d=0.02,afade=t=out:st=0.18:d=0.07,volume=0.7",
        0.25, "삑(beep)",
    ),
    "pop": (
        "sine=f=320:d=0.12:r=44100",
        "afade=t=in:st=0:d=0.005,afade=t=out:st=0.04:d=0.08,volume=1.3",
        0.12, "팝(pop)",
    ),
    "impact": (
        "sine=f=90:d=0.55:r=44100",
        "afade=t=in:st=0:d=0.01,afade=t=out:st=0.20:d=0.35,volume=2.0",
        0.55, "임팩트(impact)",
    ),
}


def make_sfx(tmpdir: str, kind: str) -> Tuple[str, float]:
    """선택한 종류의 장면전환 효과음을 ffmpeg 합성으로 생성.
    (path, 길이(초)) 반환. 'none' 이거나 실패 시 ("", 0.0)."""
    spec = SFX_SPECS.get(kind)
    if not spec or spec[0] is None:
        return "", 0.0
    lavfi_input, af, length, _ = spec
    sfx = os.path.join(tmpdir, f"sfx_{kind}.wav")
    try:
        run_ffmpeg(
            ["ffmpeg", "-y",
             "-f", "lavfi", "-i", lavfi_input,
             "-af", af,
             "-ac", "2", "-ar", "44100", sfx],
        )
        return sfx, length
    except Exception as e:
        print(f"  (전환 효과음 생성 실패, 효과음 없이 진행: {e})")
        return "", 0.0


def cut_and_concat(video_path: str, segments: List[Tuple[float, float]], out_path: str,
                   tmpdir: str, transition_style: str = "black",
                   sfx_kind: str = "whoosh", fade: float = 0.6) -> None:
    """
    Cut each segment and concatenate.
    -ss before -i  : fast keyframe seek
    re-encode      : avoids frozen frames from keyframe misalignment

    transition_style 이 'black'/'white' 이면 각 구간에 암전/화이트 화면전환을,
    sfx_kind 가 'none' 이 아니면 전환 지점 효과음을 추가한다. 클립 내부에서
    처리하므로 전체 길이/자막 타이밍은 변하지 않는다.

    각 구간은 MPEG-TS(.ts)로 인코딩한 뒤 이어붙인다. MP4를 concat -c copy
    로 이어붙이면 잘림 지점의 타임스탬프/edit-list 가 누적되어 재생 길이가
    수십 시간으로 깨지는 문제가 있어, 타임스탬프가 안전한 TS 로 처리한다.
    """
    has_video_fx = transition_style in ("black", "white")
    sfx_path, sfx_len = make_sfx(tmpdir, sfx_kind)

    segment_files = []
    n = len(segments)
    for i, (start, end) in enumerate(segments):
        seg_path = os.path.join(tmpdir, f"seg_{i:04d}.ts")
        duration = end - start

        add_sfx = bool(sfx_path) and i < n - 1   # 마지막 클립 뒤엔 효과음 없음
        do_fx = (has_video_fx or add_sfx) and duration > 1.0

        if do_fx:
            f = min(fade, duration / 4)          # 매우 짧은 클립 보호
            af = min(f, 0.15)                    # 오디오 페이드 인

            # ── 비디오 브랜치 ──
            if has_video_fx:
                color = "white" if transition_style == "white" else "black"
                vfilter = (f"[0:v]fade=t=in:st=0:d={f:.3f}:color={color},"
                           f"fade=t=out:st={duration - f:.3f}:d={f:.3f}:color={color}[v]")
                vmap = "[v]"
            else:
                vfilter = ""
                vmap = "0:v"

            # ── 오디오 브랜치 ──
            afilter_base = (f"[0:a]afade=t=in:st=0:d={af:.3f},"
                            f"afade=t=out:st={duration - f:.3f}:d={f:.3f}")
            if add_sfx:
                delay_ms = int(max(0.0, duration - sfx_len) * 1000)
                afilter = (afilter_base + "[a0];"
                           f"[1:a]adelay={delay_ms}|{delay_ms}[a1];"
                           f"[a0][a1]amix=inputs=2:duration=first:normalize=0[a]")
            else:
                afilter = afilter_base + "[a]"

            filter_complex = ";".join(p for p in (vfilter, afilter) if p)

            # -ss/-t 는 반드시 video 입력(-i video_path) '앞'에 두어
            # 해당 입력만 잘라낸다. sfx 입력 앞에 -t 가 오면 video 가
            # 안 잘리고 원본 끝까지 읽혀 파일이 수십 시간으로 깨진다.
            cmd = ["ffmpeg", "-y",
                   "-ss", str(start), "-t", str(duration), "-i", video_path]
            if add_sfx:
                cmd += ["-i", sfx_path]
            cmd += ["-filter_complex", filter_complex,
                    "-map", vmap, "-map", "[a]"]
        else:
            cmd = ["ffmpeg", "-y",
                   "-ss", str(start), "-t", str(duration), "-i", video_path]

        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-r", "30",
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                "-fps_mode", "cfr",
                "-muxpreload", "0", "-muxdelay", "0",
                "-f", "mpegts", seg_path]

        print(f"    [{i+1}/{n}] {start:.1f}s ~ {end:.1f}s 컷 중...", flush=True)
        run_ffmpeg(cmd, label=f"[{i+1}/{n}]")
        segment_files.append(seg_path)

    # TS 조각들을 concat '디먹서'로 이어붙이고 mp4 로 다시 담는다(재인코딩 X).
    # concat: 프로토콜은 각 조각의 PTS 를 그대로 누적시켜 재생 길이가
    # 실제보다 늘어난다(예: 24초 영상이 32초로 표시). 디먹서는 조각마다
    # 타임스탬프를 0 부터 재정렬해 붙이므로 길이가 정확하다.
    print("    조각들을 이어붙이는 중...", flush=True)
    list_path = os.path.join(tmpdir, "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for seg in segment_files:
            safe = seg.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    run_ffmpeg(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
         "-c", "copy", "-bsf:a", "aac_adtstoasc",
         "-movflags", "+faststart", out_path],
        label="(이어붙이기)",
    )



def safe_filename(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", title)[:80]


def main():
    parser = argparse.ArgumentParser(description="YouTube video summarizer - extracts high-energy segments")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--target-min", type=float, default=10.0, help="Target summary length in minutes (default: 10)")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--lang", default="ko", help="Language code for Whisper (default: ko)")
    parser.add_argument("--prompt", default=GAME_PROMPT,
                        help="Whisper initial_prompt (전문 용어 힌트). 기본값은 디아블로2 용어집. "
                             "빈 문자열이면 힌트 없이 받아씁니다.")
    parser.add_argument("--expand-before", type=float, default=5.0,
                        help="Seconds to expand before each energy peak (default: 5)")
    parser.add_argument("--expand-after", type=float, default=20.0,
                        help="Seconds to expand after each energy peak (default: 20)")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--max-height", type=int, default=720,
                        choices=[360, 480, 720, 1080],
                        help="Max video resolution height (default: 720)")
    parser.add_argument("--transition-style", default="black",
                        choices=list(TRANSITION_STYLES.keys()),
                        help="화면 전환 스타일: none(없음) / black(암전) / white(화이트 플래시). 기본 black")
    parser.add_argument("--sfx", dest="sfx_kind", default="whoosh",
                        choices=list(SFX_SPECS.keys()),
                        help="전환 효과음: none / whoosh(휙) / swoosh(스와이프) / "
                             "beep(삑) / pop(팝) / impact(임팩트). 기본 whoosh")
    parser.add_argument("--no-transition", dest="no_transition", action="store_true",
                        help="화면 전환 효과와 전환 효과음을 모두 끕니다 "
                             "(--transition-style none --sfx none 과 동일)")
    parser.add_argument("--bridge-gap", type=float, default=8.0,
                        help="이 시간(초) 이하로 가까운 하이라이트는 같은 내용으로 보고 "
                             "하나로 이어붙입니다 (전환 효과 없이). 기본 8초")
    args = parser.parse_args()

    if args.no_transition:
        args.transition_style = "none"
        args.sfx_kind = "none"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="summarizer_") as tmpdir:
        print(f"[1/7] Downloading video (max {args.max_height}p)...")
        video_path, title = download_video(args.url, tmpdir, max_height=args.max_height)
        print(f"  Title: {title}")
        print(f"  Saved: {video_path}")

        print(f"[2/6] Extracting audio...")
        wav_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(video_path, wav_path)

        duration = get_duration(video_path)
        print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")

        print(f"[3/6] Transcribing audio...")
        whisper_result = transcribe(wav_path, args.model, args.lang, args.prompt)
        print(f"  Transcribed {len(whisper_result.get('segments', []))} segments")

        print(f"[4/6] Analyzing audio energy...")
        energy, window_sec = compute_energy(wav_path, window_sec=0.5)

        print(f"[5/6] Finding exciting segments (target: {args.target_min} min)...")
        target_sec = int(args.target_min * 60)
        segments = find_exciting_segments(
            energy, window_sec, whisper_result,
            target_sec=target_sec,
            expand_before=args.expand_before,
            expand_after=args.expand_after,
            bridge_gap=args.bridge_gap,
        )

        if not segments:
            print("ERROR: No segments found. Try lowering --target-min or adjusting expand settings.")
            sys.exit(1)

        safe_title = safe_filename(title)
        out_video = str(output_dir / f"{safe_title}_summary.mp4")
        out_srt   = str(output_dir / f"{safe_title}_summary.srt")

        v_name = TRANSITION_STYLES.get(args.transition_style, args.transition_style)
        s_name = SFX_SPECS.get(args.sfx_kind, (None, None, 0, args.sfx_kind))[3]
        print(f"[6/6] Cutting and concatenating segments... (화면전환: {v_name} / 효과음: {s_name})")
        cut_and_concat(video_path, segments, out_video, tmpdir,
                       transition_style=args.transition_style, sfx_kind=args.sfx_kind)

        print(f"[7/7] Building subtitles...")
        srt_content = build_srt(whisper_result, segments)
        with open(out_srt, "w", encoding="utf-8") as f:
            f.write(srt_content)

        print(f"\nDone!")
        print(f"  Video : {out_video}")
        print(f"  SRT   : {out_srt}")
        print(f"\n  SRT 파일을 편집한 뒤 영상과 함께 CapCut / 편집기에 불러오세요.")


if __name__ == "__main__":
    main()

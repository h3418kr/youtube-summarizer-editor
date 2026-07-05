"""YouTube video summarizer - downloads video, finds high-energy segments, creates ~10min summary with subtitles."""
import argparse
import json
import math
import os
import re
import shutil
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
    # ffprobe/yt-dlp 는 UTF-8 로 출력한다. encoding 을 지정하지 않으면 한국어
    # Windows 기본값(cp949)으로 디코딩하다가, 경로/제목에 한글이 있으면 디코드가
    # 깨져 stdout 이 None 이 된다(로컬 한글 파일명 요약 시 발생).
    result = subprocess.run(cmd, check=True, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", **kw)
    return (result.stdout or "").strip()


def run_ffmpeg(cmd, label: str = "", cwd: str = None) -> None:
    """ffmpeg 실행: 콘솔창 숨김 + stdin 차단 + 진행상황(time=) 스트리밍.

    긴 인코딩 중에도 '멈춘 것처럼' 보이지 않도록 마지막 진행 줄을 출력한다.
    cwd 를 주면 그 폴더에서 실행한다(subtitles= 등 상대경로 필터에 사용).
    """
    # ffmpeg 는 -nostdin 으로 표준입력을 아예 건드리지 않게 한다.
    if cmd and "ffmpeg" in os.path.basename(str(cmd[0])).lower():
        cmd = [cmd[0], "-nostdin", "-hide_banner"] + list(cmd[1:])

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        cwd=cwd,
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


def get_media_size(path: str) -> Tuple[int, int]:
    """영상/이미지의 (가로, 세로) 픽셀 크기를 반환."""
    out = run(["ffprobe", "-v", "quiet", "-print_format", "json",
               "-select_streams", "v:0", "-show_streams", path])
    st = json.loads(out)["streams"][0]
    return int(st["width"]), int(st["height"])


# 워터마크(마크) 위치: key -> (사람이 읽는 이름, 코너)
#   tl=좌상단 tr=우상단 bl=좌하단 br=우하단
WM_POSITIONS = {
    "tl": "좌상단", "tr": "우상단", "bl": "좌하단", "br": "우하단",
}


def _ass_ts(sec: float) -> str:
    """초 -> ASS 시간표기 H:MM:SS.cs (센티초)."""
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs == 100:
        cs = 0
        s += 1
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    """ASS Dialogue 텍스트용 이스케이프(오버라이드 괄호/줄바꿈 처리)."""
    return (text.replace("\\", "")
                .replace("{", "(").replace("}", ")")
                .replace("\r", "").replace("\n", r"\N"))


def _build_label_ass(W: int, H: int, labels, anchor: int, x: int, y: int,
                     font: str, size: int) -> str:
    """하이라이트 소제목을 픽셀 좌표(\\pos)로 정확히 배치한 ASS 문서 생성.

    anchor : \\an 값(numpad 1~9). x, y : 텍스트 앵커의 픽셀 좌표.
    """
    head = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\n"
        "ScaledBorderAndShadow: yes\nWrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: L,{font},{size},&H00FFFFFF,&HC0000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,3,1,{anchor},10,10,10,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )
    lines = []
    for s, e, text in labels:
        body = f"{{\\an{anchor}\\pos({x},{y})}}{_ass_escape(text.strip())}"
        lines.append(f"Dialogue: 0,{_ass_ts(s)},{_ass_ts(e)},L,,0,0,0,,{body}")
    return head + "\n".join(lines) + "\n"


def apply_overlays(in_video: str, out_video: str, tmpdir: str, *,
                   watermark: str = "", wm_pos: str = "tr",
                   wm_scale: float = 0.12, wm_margin: int = 24,
                   labels=None, font: str = "Malgun Gothic",
                   label_size: int = 40) -> bool:
    """본영상에 워터마크(마크) 이미지와 하이라이트별 소제목을 새겨넣는다.

    watermark : 마크 이미지 경로(빈 문자열이면 생략)
    wm_pos    : tl/tr/bl/br (마크 위치)
    wm_scale  : 마크 가로폭 = 영상 가로폭 * wm_scale
    labels    : [(start_out, end_out, text), ...] 출력 타임라인(초) 기준 소제목.
                비면 소제목 생략. 소제목은 마크 바로 아래(마크 없으면 코너)에 뜬다.

    새겨넣을 게 있으면 out_video 로 인코딩하고 True, 없으면 아무 것도 안 하고
    False 를 반환한다(호출부에서 in_video 를 그대로 쓰면 됨).
    """
    labels = [(s, e, t) for (s, e, t) in (labels or []) if t and t.strip()]
    has_wm = bool(watermark) and os.path.isfile(watermark)
    if not has_wm and not labels:
        return False

    W, H = get_media_size(in_video)
    margin = int(wm_margin)
    top = wm_pos in ("tl", "tr")
    left = wm_pos in ("tl", "bl")

    inputs = ["-i", os.path.abspath(in_video)]
    fc_parts = []
    vlabel = "[0:v]"

    logo_h_scaled = 0
    if has_wm:
        inputs += ["-i", os.path.abspath(watermark)]
        lw = max(16, int(W * float(wm_scale)))
        try:
            iw, ih = get_media_size(watermark)
            logo_h_scaled = int(lw * ih / iw) if iw else int(lw * 0.5)
        except Exception:
            logo_h_scaled = int(lw * 0.5)
        ox = f"{margin}" if left else f"W-w-{margin}"
        oy = f"{margin}" if top else f"H-h-{margin}"
        fc_parts.append(f"[1:v]scale={lw}:-1[wm]")
        fc_parts.append(f"{vlabel}[wm]overlay={ox}:{oy}[v1]")
        vlabel = "[v1]"

    if labels:
        # 소제목을 코너에 픽셀 좌표로 배치.
        # \an(numpad): 7=상좌 8=상중 9=상우 / 1=하좌 2=하중 3=하우.
        # 마크가 이 단계에 없더라도(완성 탭에서 별도로 얹는 경우) 그 위에 마크가
        # 들어갈 여지를 남기려고 기본 여백(reserve)을 두어 코너에서 살짝 안쪽에 둔다.
        gap = 12
        reserve = logo_h_scaled if has_wm else int(H * 0.11)
        if top:
            anchor = 7 if left else 9
            y = margin + reserve + gap
        else:
            anchor = 1 if left else 3
            y = H - margin - reserve - gap
        x = margin if left else (W - margin)
        ass = _build_label_ass(W, H, labels, anchor, x, y, font, label_size)
        ass_name = "labels.ass"
        with open(os.path.join(tmpdir, ass_name), "w", encoding="utf-8") as f:
            f.write(ass)
        fc_parts.append(f"{vlabel}subtitles={ass_name}[v]")
        vmap = "[v]"
    else:
        # 워터마크만: 마지막 비디오 라벨을 그대로 출력으로.
        vmap = vlabel

    filter_complex = ";".join(fc_parts)
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", vmap, "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        "-movflags", "+faststart", os.path.abspath(out_video),
    ]
    # subtitles= 상대경로 참조를 위해 tmpdir 에서 실행.
    run_ffmpeg(cmd, label="(마크/소제목)", cwd=tmpdir)
    return True


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


def fmt_hms(sec: float, force_hours: bool = False) -> str:
    """초 -> '3:05' / '1:02:03' (챕터·구간 표기용)."""
    sec = int(round(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    if h or force_hours:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def build_chapters(segments: List[Tuple[float, float]], label: str = "하이라이트") -> str:
    """구간 목록으로 유튜브 챕터 텍스트를 만든다.

    출력 영상 타임라인 기준(누적 길이)으로 각 하이라이트의 시작 시각을 찍는다.
    전환 효과는 클립 내부에서 처리되어 전체 길이가 변하지 않으므로
    단순 누적 합이 곧 출력 타임스탬프다. 유튜브 챕터는 반드시 00:00 부터
    시작해야 하므로 첫 줄은 항상 00:00 이다.
    """
    lines = []
    t = 0.0
    for i, (s, e) in enumerate(segments, 1):
        m, sec_ = int(t // 60), int(round(t % 60))
        h = m // 60
        stamp = f"{h:d}:{m % 60:02d}:{sec_:02d}" if h else f"{m:02d}:{sec_:02d}"
        lines.append(f"{stamp} {label} {i} (원본 {fmt_hms(s, force_hours=True)})")
        t += e - s
    return "\n".join(lines) + "\n"


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
    parser.add_argument("url", help="YouTube URL 또는 로컬 영상 파일 경로")
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
    parser.add_argument("--save-video", default="",
                        help="다운로드한 원본 영상을 이 폴더에 보관합니다(삭제하지 않음). "
                             "비워두면 처리 후 원본을 삭제합니다.")
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
    parser.add_argument("--analyze-only", action="store_true",
                        help="영상을 만들지 않고 하이라이트 후보 구간만 분석해 "
                             "구간 목록 파일(_segments.txt)로 저장합니다. "
                             "자막 전사(Whisper)를 건너뛰어 훨씬 빠릅니다.")
    args = parser.parse_args()

    if args.no_transition:
        args.transition_style = "none"
        args.sfx_kind = "none"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="summarizer_") as tmpdir:
        # URL 대신 로컬 영상 파일을 주면 다운로드를 건너뛰고 그 파일을 그대로 분석한다.
        is_local = os.path.isfile(args.url)
        if is_local:
            video_path = args.url
            title = os.path.splitext(os.path.basename(args.url))[0]
            print(f"[1/6] Using local video (다운로드 건너뜀)...")
            print(f"  Title: {title}")
            print(f"  File : {video_path}")
        else:
            print(f"[1/7] Downloading video (max {args.max_height}p)...")
            video_path, title = download_video(args.url, tmpdir, max_height=args.max_height)
            print(f"  Title: {title}")
            print(f"  Saved: {video_path}")

        # 원본 영상 보관 옵션: 지정한 폴더로 복사해 두어 삭제되지 않게 한다.
        # (로컬 파일은 이미 사용자 소유이므로 다운로드한 경우에만 보관한다.)
        kept_video = None
        if args.save_video and not is_local:
            try:
                keep_dir = Path(args.save_video)
                keep_dir.mkdir(parents=True, exist_ok=True)
                kept_path = keep_dir / f"{safe_filename(title)}{os.path.splitext(video_path)[1] or '.mp4'}"
                shutil.copy2(video_path, kept_path)
                kept_video = str(kept_path)
                print(f"  원본 영상 보관: {kept_path}")
            except Exception as e:
                print(f"  (원본 영상 보관 실패, 계속 진행: {e})")

        print(f"[2/6] Extracting audio...")
        wav_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(video_path, wav_path)

        duration = get_duration(video_path)
        print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")

        if args.analyze_only:
            # 분석 전용 모드: Whisper 전사를 건너뛰어 빠르게 후보 구간만 뽑는다.
            print(f"[3/6] (분석 전용 모드 - 자막 전사 건너뜀)")
            whisper_result = {"segments": []}
        else:
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

        if args.analyze_only:
            # 후보 구간을 수동 하이라이트 탭에서 그대로 쓸 수 있는 형식으로 저장
            seg_file = output_dir / f"{safe_title}_segments.txt"
            with open(seg_file, "w", encoding="utf-8") as f:
                for s, e in segments:
                    f.write(f"{fmt_hms(s, force_hours=True)} - "
                            f"{fmt_hms(e, force_hours=True)}\n")

            # 수동 편집에 쓸 원본 영상 확보 (다운로드본이면 복사해 남긴다)
            source_video = args.url if is_local else kept_video
            if source_video is None:
                try:
                    kept_path = output_dir / (
                        f"{safe_title}{os.path.splitext(video_path)[1] or '.mp4'}")
                    shutil.copy2(video_path, kept_path)
                    source_video = str(kept_path)
                    print(f"  원본 영상 보관: {kept_path}")
                except Exception as e:
                    print(f"  (원본 영상 보관 실패: {e})")
                    source_video = ""

            print(f"\nDone! (분석 전용)")
            print(f"  후보 구간 {len(segments)}개 저장: {seg_file}")
            print(f"SEGMENTS_FILE::{seg_file}")
            print(f"SOURCE_VIDEO::{source_video}")
            print(f"\n  수동 하이라이트 탭에서 구간을 다듬은 뒤 영상을 만드세요.")
            return

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

        # 유튜브 챕터 텍스트: 설명란에 그대로 붙여넣으면 챕터가 생긴다.
        out_chapters = str(output_dir / f"{safe_title}_chapters.txt")
        with open(out_chapters, "w", encoding="utf-8") as f:
            f.write(build_chapters(segments))

        print(f"\nDone!")
        print(f"  Video    : {out_video}")
        print(f"  SRT      : {out_srt}")
        print(f"  Chapters : {out_chapters}")
        print(f"  (챕터 파일 내용을 유튜브 설명란에 붙여넣으면 구간 이동 챕터가 생깁니다.")
        print(f"   챕터는 3개 이상, 각 10초 이상일 때 유튜브에서 표시됩니다.)")
        print(f"\n  SRT 파일을 편집한 뒤 영상과 함께 CapCut / 편집기에 불러오세요.")


if __name__ == "__main__":
    main()

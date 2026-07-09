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

from faster_whisper import WhisperModel
from pydub import AudioSegment

try:
    import onnxruntime
    from PIL import Image
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


# ── Hardware acceleration encoder detection ──────────────────────────────────
_HW_ENCODER = None   # None=미탐지, ""=없음(폴백), "h264_nvenc"|"h264_qsv"|"h264_amf"
_HW_ENABLED = True


def set_hw_encoding(enabled: bool):
    """GUI/CLI에서 끄기용. 모듈 전역 _HW_ENABLED 설정."""
    global _HW_ENABLED
    _HW_ENABLED = enabled


def _detect_hw_encoder():
    """h264_nvenc -> h264_qsv -> h264_amf 순서로 0.2초 무음 테스트 인코딩을 돌려
    처음 성공하는 인코더명을 반환. 전부 실패하면 ''. 결과는 _HW_ENCODER에 캐시."""
    global _HW_ENCODER
    if _HW_ENCODER is not None:
        return _HW_ENCODER

    encoders = ["h264_nvenc", "h264_qsv", "h264_amf"]
    for encoder in encoders:
        try:
            # ffmpeg -hide_banner -f lavfi -i color=black:size=256x256:rate=30:duration=0.2 -c:v <encoder> -f null -
            cmd = [
                "ffmpeg", "-hide_banner", "-f", "lavfi",
                "-i", "color=black:size=256x256:rate=30:duration=0.2",
                "-c:v", encoder, "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, **_PROC_KW)
            if result.returncode == 0:
                _HW_ENCODER = encoder
                print(f"  GPU 인코더 감지: {encoder}")
                return _HW_ENCODER
        except Exception:
            continue

    _HW_ENCODER = ""  # 전부 실패하면 ""
    return _HW_ENCODER


def video_encode_args(crf: int) -> list:
    """하드웨어 인코더가 있으면 그에 맞는 인자, 없으면 libx264 폴백 인자를 반환."""
    if not _HW_ENABLED:
        return ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf)]

    encoder = _detect_hw_encoder()
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf)]
    elif encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", str(crf)]
    elif encoder == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp",
                "-qp_i", str(crf), "-qp_p", str(crf + 2)]
    else:
        return ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf)]


# ── Font helper for bundled fonts ────────────────────────────────────────────
def bundled_fonts_dir():
    """경로를 쓸 수 없는 환경에서 번들 폰트 폴더를 찾는다."""
    base = os.path.dirname(os.path.abspath(__file__))
    # repo root 의 배포_요약기_무설치 폴더에 fonts 가 있는 경우
    dist_fonts = os.path.join(base, "배포_요약기_무설치", "fonts")
    if os.path.isdir(dist_fonts):
        return dist_fonts
    # 배포 폴더 내에서 실행되는 경우(base 자체가 배포 폴더)
    local_fonts = os.path.join(base, "fonts")
    if os.path.isdir(local_fonts):
        return local_fonts
    return None


def copy_fonts_to(target_dir: str):
    """번들된 TTF 폰트를 대상 디렉터리에 복사한다."""
    fonts_dir = bundled_fonts_dir()
    if not fonts_dir:
        return
    target_dir = os.path.abspath(target_dir)
    if not os.path.isdir(target_dir):
        return
    try:
        for font_file in os.listdir(fonts_dir):
            if font_file.endswith(('.ttf', '.otf')):
                src = os.path.join(fonts_dir, font_file)
                dst = os.path.join(target_dir, font_file)
                if os.path.isfile(src):
                    shutil.copy(src, dst)
    except Exception:
        pass


# Windows: 하위 프로세스가 별도 콘솔(검은 창)을 띄우지 않도록.
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# 모든 하위 프로세스 공통 옵션:
#   stdin=DEVNULL      : ffmpeg/yt-dlp 가 대화형 입력을 기다리며 멈추는 것을 방지
#   creationflags      : 콘솔 창 숨김
_PROC_KW = dict(stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW)

# yt-dlp 는 번들 파이썬의 yt_dlp 패키지로 직접 호출한다.
# pip 이 만든 yt-dlp.exe 는 빌드 PC 의 파이썬 절대경로가 박혀 있어, 다른 PC 나
# 다른 경로로 압축을 풀면 실행이 깨진다. 또 python\Scripts 가 PATH 에 없으면
# 'yt-dlp' 이름 자체를 못 찾는다(FileNotFoundError). 실행 중인 파이썬
# (sys.executable) + 모듈 실행(-m yt_dlp) 이 경로·PATH 에 무관하게 항상 동작한다.
YTDLP = [sys.executable, "-m", "yt_dlp"]


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


def download_video(url: str, tmpdir: str, max_height: int = 720, cookies_browser: str = "") -> Tuple[str, str]:
    cmd_info = YTDLP + ["--print", "%(id)s|||%(title)s", "--no-playlist"]
    if cookies_browser:
        cmd_info.extend(["--cookies-from-browser", cookies_browser])
    cmd_info.append(url)
    info_raw = run(cmd_info)

    vid_id, title = info_raw.split("|||", 1)
    out_path = os.path.join(tmpdir, f"{vid_id}.mp4")
    fmt = (
        f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]"
        f"/bestvideo[height<={max_height}]+bestaudio"
        f"/best[height<={max_height}]/best"
    )
    cmd_dl = YTDLP + ["-f", fmt, "--merge-output-format", "mp4", "--newline", "-o", out_path, "--no-playlist", "--no-update"]
    if cookies_browser:
        cmd_dl.extend(["--cookies-from-browser", cookies_browser])
    cmd_dl.append(url)

    subprocess.run(cmd_dl, check=True, **_PROC_KW)
    return out_path, title


def extract_audio(video_path: str, out_wav: str) -> None:
    # aresample=async=1: 오디오 패킷 타임스탬프 사이 미세 간극을 무음으로 채워
    # wav 길이를 재생 타임라인과 일치시킨다. 점프컷처럼 수백 개 조각을 concat한
    # 영상은 간극이 누적돼 wav가 몇 초씩 짧아지고, 그 wav로 만든 자막이
    # 뒤로 갈수록 앞당겨지는(밀리는) 버그가 있었다.
    run_ffmpeg(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-af", "aresample=async=1:first_pts=0",
         "-ar", "16000", "-ac", "1", "-f", "wav", out_wav],
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
                   labels=None, font: str = "Paperlogy",
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
        # Copy bundled fonts to tmpdir so libass can find them
        copy_fonts_to(tmpdir)
        fc_parts.append(f"{vlabel}subtitles={ass_name}:fontsdir=.[v]")
        vmap = "[v]"
    else:
        # 워터마크만: 마지막 비디오 라벨을 그대로 출력으로.
        vmap = vlabel

    filter_complex = ";".join(fc_parts)
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", vmap, "-map", "0:a?",
        *video_encode_args(20),
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


def _norm_txt(s: str) -> str:
    """비교용 정규화: 한글/영문/숫자만 남기고 소문자화."""
    return re.sub(r'[^가-힣a-zA-Z0-9]', '', s).lower()


def _drop_prompt_echo(result, prompt: str):
    """initial_prompt 를 그대로 받아쓴(환각) 세그먼트를 제거한다.

    Whisper 는 음성이 불명확하면(음악·잡음·무음 등) 힌트로 준 프롬프트 문장을
    그대로 출력하는 경우가 있다(prompt echo). 세그먼트 텍스트(정규화)가 12자
    이상이면서 프롬프트(정규화)에 통째로 포함되면, 실제 말이 아니라 프롬프트
    따라읽기로 보고 버린다.
    """
    pn = _norm_txt(prompt or "")
    if not pn:
        return result
    kept = []
    for seg in result.get("segments", []):
        tn = _norm_txt(seg.get("text", ""))
        if len(tn) >= 12 and tn in pn:
            continue
        kept.append(seg)
    result["segments"] = kept
    return result


# 모듈 레벨 캐시: 모델명 -> WhisperModel 인스턴스
_model_cache = {}


def _split_segment_at_word_gaps(seg, gap_threshold=0.6):
    """세그먼트의 단어 간 큰 간격(gap_threshold초 이상)에서 분할한다.

    Args:
        seg: faster-whisper 세그먼트 (words 속성 있음)
        gap_threshold: 분할 기준 간격(초)

    Returns:
        분할된 세그먼트 dict 리스트. words가 없으면 원본 그대로 반환.
    """
    if not seg.words:
        # words가 없으면 그대로 통과
        return [{
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": [],
        }]

    # 단어 간 간격 계산해 분할점 찾기
    split_indices = []  # 분할 직후 단어의 인덱스들
    for i in range(1, len(seg.words)):
        gap = seg.words[i].start - seg.words[i-1].end
        if gap >= gap_threshold:
            split_indices.append(i)

    # 분할점이 없으면 원본 반환
    if not split_indices:
        words_list = []
        for w in seg.words:
            words_list.append({
                "word": w.word,
                "start": w.start,
                "end": w.end,
            })
        return [{
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": words_list,
        }]

    # 분할점 기준으로 단어 그룹 만들기
    result = []
    word_groups = []
    prev = 0
    for idx in split_indices:
        word_groups.append(seg.words[prev:idx])
        prev = idx
    word_groups.append(seg.words[prev:])

    # 각 그룹을 세그먼트로 변환
    for words_in_group in word_groups:
        if not words_in_group:
            continue

        # 텍스트는 word.word를 이어붙이고 strip (word는 앞에 공백 포함)
        text = "".join(w.word for w in words_in_group).strip()

        words_list = []
        for w in words_in_group:
            words_list.append({
                "word": w.word,
                "start": w.start,
                "end": w.end,
            })

        result.append({
            "start": words_in_group[0].start,
            "end": words_in_group[-1].end,
            "text": text,
            "words": words_list,
        })

    return result


def transcribe(wav_path: str, model_name: str, lang: str, prompt: str = None):
    global _model_cache

    # 모델명 정규화: "large" -> "large-v3"
    normalized_model = "large-v3" if model_name == "large" else model_name

    print(f"  Transcribing with faster-whisper ({normalized_model})...")

    # 캐시에서 모델 로드 또는 새로 생성
    if normalized_model not in _model_cache:
        _model_cache[normalized_model] = WhisperModel(
            normalized_model,
            device="cpu",
            compute_type="int8",
            download_root=_bundled_model_root()
        )

    model = _model_cache[normalized_model]

    def _run(p):
        # faster-whisper는 제너레이터와 info를 반환한다
        segments_gen, info = model.transcribe(
            wav_path,
            language=lang,
            word_timestamps=True,
            initial_prompt=p or None,
            condition_on_previous_text=False,
            vad_filter=True,
        )

        # 제너레이터를 리스트로 변환하고, 기존 형식으로 어댑트
        segments_list = []
        full_text = []

        for seg in segments_gen:
            # 단어 간 큰 간격에서 세그먼트 분할
            split_segs = _split_segment_at_word_gaps(seg, gap_threshold=0.6)

            for split_seg in split_segs:
                segments_list.append(split_seg)
                full_text.append(split_seg["text"])

        # 분할 후 최종 id 재부여
        for i, seg in enumerate(segments_list):
            seg["id"] = i

        return {
            "segments": segments_list,
            "language": info.language,
            "text": " ".join(full_text),
        }

    result = _run(prompt)
    if prompt:
        # _drop_prompt_echo 는 result 를 제자리 수정하므로, 필터 전에 세그먼트가
        # 있었는지 먼저 기록해 둔다.
        had_segments = bool(result.get("segments"))
        filtered = _drop_prompt_echo(result, prompt)
        # 프롬프트를 통째로 따라읽어(환각) 세그먼트가 전부 걸러졌다면, 실제
        # 인식이 실패한 것이다. 이땐 용어 힌트 없이 한 번 더 인식해 자막을 살린다.
        # (힌트가 없으니 전문 용어 정확도는 떨어질 수 있으나 자막은 나온다.)
        if had_segments and not filtered.get("segments"):
            print("  (용어 힌트 따라읽기 감지 - 힌트 없이 다시 인식합니다)")
            result = _run(None)
        else:
            result = filtered
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


def compute_voice_energy(wav_path: str, tmpdir: str, window_sec: float = 0.5) -> Tuple[np.ndarray, float]:
    """목소리 대역(200-3800Hz) 에너지 곡선. 반환: (np.ndarray, window_sec)

    방송인 음성이 게임 효과음·음악보다 우선되도록 하는 목적.
    """
    try:
        # ffmpeg로 voice.wav 생성 (200-3800Hz 필터)
        voice_wav = os.path.join(tmpdir, "voice_filtered.wav")
        cmd = [
            "ffmpeg", "-y", "-i", wav_path,
            "-af", "highpass=f=200,lowpass=f=3800",
            "-ar", "16000", "-ac", "1", voice_wav
        ]
        run_ffmpeg(cmd, label="(목소리대역-필터)")

        # voice.wav에서 에너지 계산
        audio = AudioSegment.from_wav(voice_wav)
        sr = audio.frame_rate
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        window = int(sr * window_sec)
        n_windows = len(samples) // window
        energy = np.zeros(n_windows)
        for i in range(n_windows):
            chunk = samples[i * window:(i + 1) * window]
            energy[i] = np.sqrt(np.mean(chunk ** 2))
        return energy, window_sec

    except Exception as e:
        print(f"  (목소리대역 에너지 계산 실패: {e})")
        return np.array([], dtype=np.float32), window_sec


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    kernel_size = int(6 * sigma) | 1  # ensure odd
    half = kernel_size // 2
    k = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same")


def compute_chat_activity(video_path: str, chat_region: Tuple[int, int, int, int],
                          duration: float, tmpdir: str, sample_sec: float = 2.0) -> Tuple[np.ndarray, float, float, float]:
    """Improved chat activity curve using masked pixel approach.

    Focus on text-like pixels (bright, contrasty) to reduce game background noise.

    Returns: (curve, sample_sec, active_ratio, chat_weight)
    """
    try:
        x, y, w, h = chat_region
        W, H = get_media_size(video_path)

        crop_str = f"{w}:{h}:{x}:{y}"
        frame_dir = os.path.join(tmpdir, "chat_activity_frames_v2")
        os.makedirs(frame_dir, exist_ok=True)

        # Extract frames in RGB (for better text detection)
        scale_h = 120
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"fps=1/{sample_sec},crop={crop_str},scale=240:-2,format=rgb24",
            os.path.join(frame_dir, "frame_%05d.png")
        ]
        run_ffmpeg(cmd, label="(채팅활동-프레임추출)")

        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
        if len(frame_files) < 2:
            return np.array([], dtype=np.float32), sample_sec, 0.0, 0.0

        from PIL import Image as PILImage

        activity = []
        active_threshold = 3.0 / 255.0

        for i in range(len(frame_files) - 1):
            try:
                img1 = np.array(PILImage.open(os.path.join(frame_dir, frame_files[i])), dtype=np.float32)
                img2 = np.array(PILImage.open(os.path.join(frame_dir, frame_files[i + 1])), dtype=np.float32)

                # Convert RGB to grayscale for analysis
                if len(img1.shape) == 3:
                    img1 = np.mean(img1, axis=2)
                    img2 = np.mean(img2, axis=2)

                # Masked change: focus on bright, contrasty pixels
                # Text pixels are typically bright and show change between frames
                f1 = img1 / 255.0
                f2 = img2 / 255.0

                bright_mask = np.maximum(f1, f2) > (150 / 255.0)
                contrast_mask = np.abs(f1 - f2) > (10 / 255.0)
                text_mask = bright_mask & contrast_mask

                if np.sum(text_mask) > 0:
                    # Only measure change in text-like pixels
                    masked_change = np.mean(np.abs(f1[text_mask] - f2[text_mask]))
                else:
                    # Fallback: raw change (lower weight)
                    masked_change = np.mean(np.abs(f1 - f2)) * 0.5

                activity.append(masked_change)
            except:
                activity.append(0.0)

        activity = np.array(activity, dtype=np.float32)

        # 3-point smoothing
        if len(activity) >= 3:
            smoothed = np.convolve(activity, np.array([1, 1, 1]) / 3.0, mode='same')
        else:
            smoothed = activity

        # Compute metrics
        active_ratio = np.sum(activity >= active_threshold) / len(activity) if len(activity) > 0 else 0.0
        chat_weight = min(1.0, active_ratio / 0.2)

        return smoothed, sample_sec, active_ratio, chat_weight

    except Exception as e:
        print(f"  (채팅활동 곡선 계산 실패: {e})")
        return np.array([], dtype=np.float32), sample_sec, 0.0, 0.0


def find_exciting_segments(
    energy: np.ndarray,
    window_sec: float,
    whisper_result,
    target_sec: int = 600,
    expand_before: float = 5.0,
    expand_after: float = 20.0,
    bridge_gap: float = 8.0,
    chat_curve: np.ndarray = None,
    chat_sample_sec: float = None,
    chat_weight: float = None,
    voice_energy: np.ndarray = None,
) -> List[Tuple[float, float]]:
    """Find high-energy segments that sum to approximately target_sec.

    bridge_gap: 선택된 하이라이트끼리 원본상 시간차가 이 값(초) 이하이면
                같은 내용으로 보고 사이 구간까지 포함해 하나로 이어붙인다.
                (전환 효과는 이렇게 병합된 최종 구간들 '사이'에만 들어간다.)
    chat_curve: 채팅 활동 곡선 (없으면 오디오 에너지만 사용)
    chat_sample_sec: 채팅 곡선의 샘플 간격(초)
    chat_weight: 채팅 적응 가중치 (min(1.0, active_ratio / 0.2))
    voice_energy: 목소리 대역 에너지 (200-3800Hz 필터링)
    """
    sigma = 10.0 / window_sec  # smooth over ~10s
    smoothed = _gaussian_smooth(energy, sigma)

    # z-score 정규화: 전체 에너지
    energy_mean = np.mean(smoothed)
    energy_std = np.std(smoothed)
    if energy_std > 1e-6:
        energy_z = (smoothed - energy_mean) / energy_std
    else:
        energy_z = np.zeros_like(smoothed)

    # z-score 정규화: 목소리 대역 에너지 (항상 계산)
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
        print(f"  채팅 가중치: {chat_weight:.2f} (활동 비율 {chat_weight/1.0*20:.0f}%+)")
        print("  (목소리 반응 + 채팅 반응 반영된 점수로 분석 중)")
    else:
        # 채팅이 없어도 목소리는 적용
        smoothed = 0.5 * energy_z + 1.0 * voice_z
        if len(voice_z) > 0 and np.any(voice_z != 0):
            print("  (목소리 반응 반영된 점수로 분석 중)")

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
    "closeup": "클로즈업 (캠 확대)",
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


def _detect_faces_ultraface(sess, input_name, frame_files, frame_dir, W, H):
    """UltraFace 모델로 얼굴 감지."""
    all_boxes = []
    for frame_idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, frame_file)
        img = Image.open(frame_path).convert('RGB')
        model_w, model_h = 320, 240
        img_resized = img.resize((model_w, model_h))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = (img_array - 127.0) / 128.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, 0)
        outputs = sess.run(None, {input_name: img_array})
        scores = outputs[0]
        boxes = outputs[1]
        scores_reshaped = scores[0, :, 1]
        for j in range(len(scores_reshaped)):
            if scores_reshaped[j] > 0.7:
                x1, y1, x2, y2 = boxes[0, j]
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
                all_boxes.append((x1, y1, x2, y2, float(scores_reshaped[j]), frame_idx))
    return all_boxes


def _detect_faces_anime(sess, input_name, frame_files, frame_dir, W, H):
    """Anime face YOLO 모델로 얼굴 감지. 출력: (1, 5, 8400) -> [x,y,w,h,conf] x 8400"""
    all_boxes = []
    yolo_size = 640
    for frame_idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, frame_file)
        img = Image.open(frame_path).convert('RGB')
        orig_w, orig_h = img.size
        scale = min(yolo_size / orig_w, yolo_size / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_array = np.full((yolo_size, yolo_size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (yolo_size - new_w) // 2, (yolo_size - new_h) // 2
        img_array[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = np.array(img_resized)
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, 0)
        outputs = sess.run(None, {input_name: img_array})
        # 출력 shape: (1, 5, 8400) -> [x, y, w, h, conf]
        pred = outputs[0][0]  # (5, 8400)
        for i in range(pred.shape[1]):
            conf = pred[4, i]
            if conf > 0.15:
                x, y, w, h = pred[0, i], pred[1, i], pred[2, i], pred[3, i]
                # letterbox 역변환
                x, y = (x - pad_x) / scale, (y - pad_y) / scale
                w, h = w / scale, h / scale
                x1, y1, x2, y2 = int(max(0, x - w/2)), int(max(0, y - h/2)), int(min(W, x + w/2)), int(min(H, y + h/2))
                if x1 < x2 and y1 < y2:
                    all_boxes.append((x1, y1, x2, y2, float(conf), frame_idx))
    return all_boxes


def _cluster_and_crop_faces(all_boxes, frame_idx, W, H, expand_scale_w=2.2, expand_scale_h=2.0):
    """감지된 얼굴 박스를 클러스터링하고 크롭 영역 계산."""
    from collections import defaultdict

    def nms_boxes_per_frame(boxes, iou_thresh=0.4):
        if not boxes:
            return []
        frames_dict = defaultdict(list)
        for box in boxes:
            frame_idx = box[5] if len(box) >= 6 else 0
            frames_dict[frame_idx].append(box)
        keep = []
        for frame_id in sorted(frames_dict.keys()):
            frame_boxes = frames_dict[frame_id]
            frame_boxes_sorted = sorted(frame_boxes, key=lambda b: -b[4])
            frame_keep = []
            for box in frame_boxes_sorted:
                keep_this = True
                for kept_box in frame_keep:
                    x1a, y1a, x2a, y2a = box[:4]
                    x1b, y1b, x2b, y2b = kept_box[:4]
                    inter_x1, inter_y1 = max(x1a, x1b), max(y1a, y1b)
                    inter_x2, inter_y2 = min(x2a, x2b), min(y2a, y2b)
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        area_a, area_b = (x2a - x1a) * (y2a - y1a), (x2b - x1b) * (y2b - y1b)
                        union_area = area_a + area_b - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0
                        if iou > iou_thresh:
                            keep_this = False
                            break
                if keep_this:
                    frame_keep.append(box)
            keep.extend(frame_keep)
        return keep

    nms_boxes_result = nms_boxes_per_frame(all_boxes)
    if not nms_boxes_result:
        return None

    clusters, assigned = [], set()
    for i, box in enumerate(nms_boxes_result):
        if i in assigned:
            continue
        x1, y1, x2, y2 = box[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cluster = [i]
        assigned.add(i)
        threshold = W * 0.08
        for j in range(i + 1, len(nms_boxes_result)):
            if j in assigned:
                continue
            x1j, y1j, x2j, y2j = nms_boxes_result[j][:4]
            cxj, cyj = (x1j + x2j) / 2, (y1j + y2j) / 2
            dist = ((cx - cxj) ** 2 + (cy - cyj) ** 2) ** 0.5
            if dist <= threshold:
                cluster.append(j)
                assigned.add(j)
        clusters.append(cluster)

    cluster_info = []
    for cluster_indices in clusters:
        unique_frames = set()
        for idx in cluster_indices:
            if len(nms_boxes_result[idx]) >= 6:
                unique_frames.add(nms_boxes_result[idx][5])
        cluster_info.append({'indices': cluster_indices, 'frame_count': len(unique_frames), 'unique_frames': unique_frames})

    sample_frame_count = frame_idx + 1
    threshold_frames = int(sample_frame_count * 0.6)
    qualified = [c for c in cluster_info if c['frame_count'] >= threshold_frames]

    if not qualified:
        return None

    best_cluster = max(qualified, key=lambda c: len(c['indices']))
    selected_boxes = np.array([nms_boxes_result[idx][:4] for idx in best_cluster['indices']], dtype=np.float32)
    face_x1, face_y1, face_x2, face_y2 = int(np.median(selected_boxes[:, 0])), int(np.median(selected_boxes[:, 1])), int(np.median(selected_boxes[:, 2])), int(np.median(selected_boxes[:, 3]))
    face_w, face_h = face_x2 - face_x1, face_y2 - face_y1

    crop_w, crop_h = int(face_w * expand_scale_w), int(face_h * expand_scale_h)
    crop_x, crop_y = face_x1 + (face_w - crop_w) // 2, face_y1 - int(crop_h * 0.18)
    crop_x, crop_y = max(0, min(crop_x, W - crop_w)), max(0, min(crop_y, H - crop_h))

    aspect_ratio = 16.0 / 9.0
    current_ratio = crop_w / crop_h if crop_h > 0 else 1
    if current_ratio < aspect_ratio:
        new_w = int(crop_h * aspect_ratio)
        crop_x -= (new_w - crop_w) // 2
        crop_w = new_w
    else:
        new_h = int(crop_w / aspect_ratio)
        crop_y -= (new_h - crop_h) // 2
        crop_h = new_h

    crop_x, crop_y = max(0, min(crop_x, W - crop_w)), max(0, min(crop_y, H - crop_h))
    crop_w, crop_h = min(crop_w, W - crop_x), min(crop_h, H - crop_y)

    inset_w, inset_h = int(crop_w * 0.92), int(crop_h * 0.92)
    crop_x += (crop_w - inset_w) // 2
    crop_y += (crop_h - inset_h) // 2
    crop_w, crop_h = inset_w, inset_h

    return (crop_x, crop_y, crop_w, crop_h)


def detect_chat_region(video_path: str, tmpdir: str, exclude_region=None):
    """Improved chat region detection using color saturation heuristic.

    Strategy:
    1. Sample 12 time points (5%-95% of video)
    2. Extract RGB frames from left/right bands
    3. Measure color saturation (indicator of colored nicknames)
    4. Left band with high saturation typically has chat
    5. Conservative auto-detection: left_sat > max(0.25, right_sat*1.3) or right_sat > max(0.25, left_sat*1.3)

    Returns: (x, y, w, h) or None if detection fails
    """
    try:
        W, H = get_media_size(video_path)
        dur = get_duration(video_path)

        # Sample 12 time points
        start_t, end_t = dur * 0.05, dur * 0.95
        sample_times = [start_t + (end_t - start_t) * i / 11 for i in range(12)]

        band_w_scaled = 160

        def evaluate_band_saturation(band_name, x_range_normalized):
            """Measure average saturation in band."""
            saturation_values = []

            for center_t in sample_times:
                t1 = max(0, center_t - 0.5)
                t2 = min(dur, center_t + 0.5)

                for t in [t1, t2]:
                    try:
                        crop_x1 = int(W * x_range_normalized[0])
                        crop_x2 = int(W * x_range_normalized[1])
                        crop_w = crop_x2 - crop_x1

                        # Extract RGB
                        cmd = [
                            "ffmpeg", "-y", "-ss", str(t), "-i", video_path,
                            "-vf", f"crop={crop_w}:{H}:{crop_x1}:0,scale={band_w_scaled}:-1,format=rgb24",
                            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
                        ]
                        proc = subprocess.Popen(
                            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL, creationflags=0x08000000 if os.name == "nt" else 0
                        )
                        raw_data, _ = proc.communicate(timeout=10)

                        actual_h = len(raw_data) // (band_w_scaled * 3)
                        if actual_h > 0:
                            frame = np.frombuffer(raw_data, dtype=np.uint8).reshape(actual_h, band_w_scaled, 3)

                            # Calculate saturation
                            r = frame[..., 0].astype(float) / 255.0
                            g = frame[..., 1].astype(float) / 255.0
                            b = frame[..., 2].astype(float) / 255.0

                            max_c = np.maximum(np.maximum(r, g), b)
                            min_c = np.minimum(np.minimum(r, g), b)
                            sat = (max_c - min_c) / (max_c + 1e-6)

                            # Measure high-saturation pixels
                            high_sat_ratio = np.sum(sat > 0.3) / sat.size
                            saturation_values.append(high_sat_ratio)
                    except:
                        pass

            avg_sat = np.mean(saturation_values) if saturation_values else 0
            return avg_sat

        # Compare left vs right
        left_sat = evaluate_band_saturation("left", (0, 1/3))
        right_sat = evaluate_band_saturation("right", (2/3, 1))

        print(f"  Chat detection: Left saturation={left_sat:.2%}, Right={right_sat:.2%}")

        # Conservative auto-detection
        if left_sat > max(0.25, right_sat * 1.3):
            x_range = (0, 1/3)
            selected_band = "left"
        elif right_sat > max(0.25, left_sat * 1.3):
            x_range = (2/3, 1)
            selected_band = "right"
        else:
            print(f"  (채팅창 자동 판별 불확실 - 채팅 미사용. '채팅 위치'를 왼쪽/오른쪽으로 직접 지정하면 사용됩니다)")
            return None

        # Refine vertical bounds using grayscale activity
        crop_x1 = int(W * x_range[0])
        crop_x2 = int(W * x_range[1])
        crop_w = crop_x2 - crop_x1

        row_activity = None
        for center_t in sample_times[:3]:
            t1 = max(0, center_t - 0.5)
            t2 = min(dur, center_t + 0.5)

            frames_pair = []
            for t in [t1, t2]:
                try:
                    cmd = [
                        "ffmpeg", "-y", "-ss", str(t), "-i", video_path,
                        "-vf", f"crop={crop_w}:{H}:{crop_x1}:0,scale={band_w_scaled}:-1,format=gray",
                        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray8", "-"
                    ]
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL, creationflags=0x08000000 if os.name == "nt" else 0
                    )
                    raw_data, _ = proc.communicate(timeout=10)
                    actual_h = len(raw_data) // band_w_scaled
                    if actual_h > 0:
                        frame_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(actual_h, band_w_scaled)
                        frames_pair.append(frame_array.astype(np.float32))
                except:
                    pass

            if len(frames_pair) == 2:
                diff = np.abs(frames_pair[0] - frames_pair[1])
                row_means = np.mean(diff, axis=1)
                if row_activity is None:
                    row_activity = row_means
                else:
                    row_activity += row_means

        if row_activity is None:
            # Fallback: use full height
            y_start, y_end = 0, H
        else:
            # Find continuous active region
            threshold = np.mean(row_activity) * 0.5
            active_rows = np.where(row_activity > threshold)[0]

            if len(active_rows) > 0:
                y_start, y_end = active_rows[0], active_rows[-1]
                # 10% margin removal
                band_h = y_end - y_start + 1
                margin = max(1, int(band_h * 0.1))
                y_start = min(y_start + margin, y_end - margin)
            else:
                y_start, y_end = 0, H

        # Convert to original coordinates
        scale_h = len(row_activity) if row_activity is not None else H
        chat_y_scaled = int(y_start * H / scale_h) if scale_h > 0 else 0
        chat_h_scaled = int((y_end - y_start) * H / scale_h) if scale_h > 0 else H

        chat_x = crop_x1
        chat_y = chat_y_scaled
        chat_w = crop_w
        chat_h = chat_h_scaled

        # Bounds check
        chat_x = max(0, min(chat_x, W - chat_w))
        chat_y = max(0, min(chat_y, H - chat_h))
        chat_w = min(chat_w, W - chat_x)
        chat_h = min(chat_h, H - chat_y)

        print(f"  Chat region detected: x={chat_x}, y={chat_y}, {chat_w}x{chat_h} ({selected_band})")
        return (chat_x, chat_y, chat_w, chat_h)

    except Exception as e:
        print(f"  (Chat detection error: {e})")
        return None


def detect_cam_region(video_path: str, tmpdir: str):
    """영상에서 고정 캠(얼굴) 위치를 자동 감지. UltraFace 후 애니 모델 시도."""
    if not _ONNX_AVAILABLE:
        return None

    base = os.path.dirname(os.path.abspath(__file__))
    ultraface_paths = [os.path.join(base, "models", "ultraface-rfb-320.onnx"), os.path.join(base, "assets", "ultraface-rfb-320.onnx")]
    anime_model_paths = [os.path.join(base, "assets", "animeface.onnx"), os.path.join(base, "models", "animeface.onnx")]

    ultraface_path = next((p for p in ultraface_paths if os.path.isfile(p)), None)
    anime_model_path = next((p for p in anime_model_paths if os.path.isfile(p)), None)

    if not ultraface_path and not anime_model_path:
        print("  (얼굴 감지 모델 없음, 캠 자동 감지 비활성화)")
        return None

    try:
        W, H = get_media_size(video_path)
        dur = get_duration(video_path)
        start_t, end_t = dur * 0.05, dur * 0.95

        frame_dir = os.path.join(tmpdir, "face_detect_frames")
        os.makedirs(frame_dir, exist_ok=True)

        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{(end_t - start_t) / 11})',setpts=PTS-STARTPTS",
            "-vsync", "vfr",
            os.path.join(frame_dir, "frame_%04d.png")
        ]
        run_ffmpeg(cmd, label="(얼굴감지-프레임추출)")

        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
        if not frame_files:
            return None

        # 1단계: UltraFace 시도
        if ultraface_path:
            try:
                sess = onnxruntime.InferenceSession(ultraface_path, providers=['CPUExecutionProvider'])
                input_name = sess.get_inputs()[0].name
                all_boxes = _detect_faces_ultraface(sess, input_name, frame_files, frame_dir, W, H)
                if all_boxes:
                    result = _cluster_and_crop_faces(all_boxes, len(frame_files) - 1, W, H, expand_scale_w=2.2, expand_scale_h=2.0)
                    if result:
                        crop_x, crop_y, crop_w, crop_h = result
                        print(f"  캠 자동 감지: x={crop_x}, y={crop_y}, {crop_w}x{crop_h}")
                        return result
            except Exception as e:
                print(f"  (UltraFace 오류: {e})")

        # 2단계: 애니 모델 시도
        if anime_model_path:
            try:
                sess = onnxruntime.InferenceSession(anime_model_path, providers=['CPUExecutionProvider'])
                input_name = sess.get_inputs()[0].name
                all_boxes = _detect_faces_anime(sess, input_name, frame_files, frame_dir, W, H)
                if all_boxes:
                    result = _cluster_and_crop_faces(all_boxes, len(frame_files) - 1, W, H, expand_scale_w=1.6, expand_scale_h=1.6)
                    if result:
                        crop_x, crop_y, crop_w, crop_h = result
                        print(f"  캠 자동 감지(버튜버): x={crop_x}, y={crop_y}, {crop_w}x{crop_h}")
                        return result
            except Exception as e:
                print(f"  (애니 얼굴 감지 오류: {e})")

        print("  (고정 캠 판정 실패 - 얼굴이 일정 위치에 반복 등장하지 않음)")
        return None

    except Exception as e:
        print(f"  (자동 감지 오류: {e})")
        return None


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
                   sfx_kind: str = "whoosh", fade: float = 0.6,
                   cam_region: str = "br", closeup_sec: float = 1.5,
                   closeup_every: int = 1, punchins: dict = None) -> None:
    """
    Cut each segment and concatenate.
    -ss before -i  : fast keyframe seek
    re-encode      : avoids frozen frames from keyframe misalignment

    transition_style 이 'black'/'white' 이면 각 구간에 암전/화이트 화면전환을,
    'closeup' 이면 구간 끝부분을 캠 영역으로 crop후 확대해 브리지 영상을 삽입한다.
    sfx_kind 가 'none' 이 아니면 전환 지점 효과음을 추가한다. 클립 내부에서
    처리하므로 전체 길이/자막 타이밍은 변하지 않는다.

    각 구간은 MPEG-TS(.ts)로 인코딩한 뒤 이어붙인다. MP4를 concat -c copy
    로 이어붙이면 잘림 지점의 타임스탬프/edit-list 가 누적되어 재생 길이가
    수십 시간으로 깨지는 문제가 있어, 타임스탬프가 안전한 TS 로 처리한다.

    cam_region: "br"(기본) 등 프리셋 또는 "x,y,w,h" 픽셀 문자열 (closeup/punchins일 때만 사용)
    closeup_sec: closeup 브리지/punchins 길이 (기본 1.5초)
    closeup_every: closeup 브리지를 몇 번의 전환마다 넣을지 (기본 1=매번)
    punchins: {구간인덱스: [펀치인시작초, ...]} dict, 각 구간에서 펀치인 시작 시점(들) 저장. 리스트 형식으로 다중 지원
    """
    has_video_fx = transition_style in ("black", "white", "closeup")
    sfx_path, sfx_len = make_sfx(tmpdir, sfx_kind)
    punchins = punchins or {}

    # closeup 브리지/punchins 캠 영역을 위한 해상도 취득
    W, H = None, None
    need_cam_region = (transition_style == "closeup") or bool(punchins)
    if need_cam_region:
        try:
            W, H = get_media_size(video_path)
        except Exception:
            print(f"  (해상도 취득 실패, closeup/punchins 비활성화)")
            if transition_style == "closeup":
                transition_style = "none"
                has_video_fx = False
            punchins = {}

    # "auto" 감지
    if cam_region == "auto" and need_cam_region:
        try:
            detected = detect_cam_region(video_path, tmpdir)
            if detected:
                cam_region = ",".join(str(v) for v in detected)
            else:
                print("  (자동 감지 실패 - 우하단 프리셋 사용)")
                cam_region = "br"
        except Exception as e:
            print(f"  (자동 감지 오류: {e} - 우하단 프리셋 사용)")
            cam_region = "br"

    # closeup 캠 영역 계산
    def calc_cam_region(cam_region_str: str) -> Tuple[int, int, int, int]:
        """cam_region 프리셋 또는 "x,y,w,h" 파싱 -> (x, y, w, h)"""
        if "," in cam_region_str:
            try:
                parts = [int(p.strip()) for p in cam_region_str.split(",")]
                if len(parts) == 4:
                    return tuple(parts)
            except ValueError:
                pass
        # 프리셋: 캠 박스 크기 W*0.25, H*0.25 + 실제 크롭은 그 중앙 84% (모서리 잡음 흡수)
        # (closeup 전환뿐 아니라 펀치인 단독 사용 시에도 프리셋이 동작해야 한다)
        if W and H:
            box_w = int(W * 0.25)
            box_h = int(H * 0.25)
            cw = int(box_w * 0.84)  # 실제 크롭 폭
            ch = int(box_h * 0.84)  # 실제 크롭 높이
            # 각 프리셋 위치의 박스 정의
            if cam_region_str == "tl":
                box_x, box_y = 0, 0
            elif cam_region_str == "tr":
                box_x, box_y = W - box_w, 0
            elif cam_region_str == "bl":
                box_x, box_y = 0, H - box_h
            else:  # "br" 또는 기본값
                box_x, box_y = W - box_w, H - box_h
            # 박스 내에서 중앙 정렬
            x = box_x + (box_w - cw) // 2
            y = box_y + (box_h - ch) // 2
            return (max(0, x), max(0, y), cw, ch)
        # 기본 fallback (25% 기준)
        fallback_w = max(1, int(W * 0.25) if W else 320)
        fallback_h = max(1, int(H * 0.25) if H else 240)
        return (0, 0, fallback_w, fallback_h)

    segment_files = []
    n = len(segments)
    for i, (start, end) in enumerate(segments):
        duration = end - start

        # closeup 브리지 + punchins 로직: 구간을 최대 4조각으로 분할
        # 각 part는 (시작시각, 종료시각, 타입문자열)
        # 타입: "normal"(일반), "bridge"(closeup 브리지), "punchin_bridge"(펀치인 캠크롭)
        is_last = (i == n - 1)
        parts_to_encode = []

        # closeup 브리지 여부 (closeup_every 조건)
        do_closeup_bridge = (
            transition_style == "closeup" and not is_last and
            duration >= closeup_sec * 3 and i % closeup_every == 0
        )

        # 펀치인 여부 (리스트 형식 다중 지원)
        punchin_times = punchins.get(i, [])
        has_punchin = len(punchin_times) > 0

        if do_closeup_bridge or has_punchin:
            # 펀치인과 closeup 브리지 모두 있을 수 있음
            if has_punchin:
                # 펀치인 시간 유효성 검사 및 필터링
                # 1. [start+1, end-closeup_sec-(브리지 있으면 closeup_sec+1, 없으면 1)] 범위
                min_punchin = start + 1
                max_punchin = end - closeup_sec - (closeup_sec + 1 if do_closeup_bridge else 1)

                valid_punchins = []
                for pt in punchin_times:
                    if pt < min_punchin or pt > max_punchin:
                        print(f"  (펀치인 {pt:.1f}s - 구간{i} 범위 초과 [{min_punchin:.1f}, {max_punchin:.1f}] 제외)")
                        continue
                    valid_punchins.append(pt)

                # 2. 인접 펀치인과 간격이 closeup_sec+1 미만이면 뒤의 것 제외
                filtered_punchins = []
                for idx, pt in enumerate(valid_punchins):
                    skip = False
                    for prev_pt in filtered_punchins:
                        if pt - prev_pt < closeup_sec + 1:
                            print(f"  (펀치인 {pt:.1f}s - 직전 펀치인과 간격 부족({pt - prev_pt:.1f}s < {closeup_sec + 1:.1f}s) 제외)")
                            skip = True
                            break
                    if not skip:
                        filtered_punchins.append(pt)

                # 유효한 펀치인들로 구간 분할: [normal][punch][normal][punch]...[normal(+bridge)]
                if filtered_punchins:
                    curr_time = start
                    for pidx, pt in enumerate(filtered_punchins):
                        punchin_end = pt + closeup_sec
                        if punchin_end > end:
                            punchin_end = end

                        # normal part (if gap exists)
                        if pt > curr_time:
                            parts_to_encode.append((curr_time, pt, "normal"))

                        # punchin part
                        parts_to_encode.append((pt, punchin_end, "punchin_bridge"))
                        curr_time = punchin_end

                    # 남은 부분 + closeup 브리지
                    if curr_time < end:
                        if do_closeup_bridge and (end - curr_time) >= closeup_sec:
                            parts_to_encode.append((curr_time, end - closeup_sec, "normal"))
                            parts_to_encode.append((end - closeup_sec, end, "bridge"))
                        else:
                            parts_to_encode.append((curr_time, end, "normal"))
                else:
                    # 유효한 펀치인이 없으면 일반 처리
                    if do_closeup_bridge:
                        parts_to_encode.append((start, end - closeup_sec, "normal"))
                        parts_to_encode.append((end - closeup_sec, end, "bridge"))
                    else:
                        parts_to_encode.append((start, end, "normal"))
            elif do_closeup_bridge:
                # closeup 브리지만 있음
                parts_to_encode.append((start, end - closeup_sec, "normal"))
                parts_to_encode.append((end - closeup_sec, end, "bridge"))
        else:
            # 일반 인코딩
            parts_to_encode.append((start, end, "normal"))

        for part_idx, (part_start, part_end, part_type) in enumerate(parts_to_encode):
            # seg_path 결정
            if part_type == "bridge":
                seg_path = os.path.join(tmpdir, f"seg_{i:04d}_bridge.ts")
            elif part_type == "punchin_bridge":
                # 한 구간에 펀치인이 여러 개일 수 있으므로 part_idx로 파일명을 구분
                # (구분하지 않으면 뒤 펀치인이 앞 파일을 덮어써 같은 장면이 반복된다)
                seg_path = os.path.join(tmpdir, f"seg_{i:04d}_punchin{part_idx}.ts")
            else:
                seg_path = os.path.join(tmpdir, f"seg_{i:04d}.ts") if part_idx == 0 else os.path.join(tmpdir, f"seg_{i:04d}_part{part_idx}.ts")

            part_duration = part_end - part_start
            add_sfx = bool(sfx_path) and i < n - 1 and part_type == "bridge"  # closeup 브리지 끝에만 효과음
            do_fx = (has_video_fx or add_sfx) and part_duration > 1.0

            if do_fx:
                f = min(fade, part_duration / 4)    # 매우 짧은 클립 보호
                af = min(f, 0.15)                   # 오디오 페이드 인

                # ── 비디오 브랜치 ──
                if part_type in ("bridge", "punchin_bridge"):
                    # 캠 영역 crop + 확대 (closeup 브리지, 펀치인 모두 동일)
                    cx, cy, cw, ch = calc_cam_region(cam_region)
                    # 입력이 1080p 기준이 아니면 W:H로 출력
                    out_w, out_h = W, H
                    vfilter = (f"[0:v]crop={cw}:{ch}:{cx}:{cy},"
                               f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                               f"crop={out_w}:{out_h}[v]")
                    vmap = "[v]"
                elif transition_style in ("black", "white"):
                    color = "white" if transition_style == "white" else "black"
                    vfilter = (f"[0:v]fade=t=in:st=0:d={f:.3f}:color={color},"
                               f"fade=t=out:st={part_duration - f:.3f}:d={f:.3f}:color={color}[v]")
                    vmap = "[v]"
                else:
                    vfilter = ""
                    vmap = "0:v"

                # ── 오디오 브랜치 ──
                afilter_base = (f"[0:a]afade=t=in:st=0:d={af:.3f},"
                                f"afade=t=out:st={part_duration - f:.3f}:d={f:.3f}")
                if add_sfx:
                    delay_ms = int(max(0.0, part_duration - sfx_len) * 1000)
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
                       "-ss", str(part_start), "-t", str(part_duration), "-i", video_path]
                if add_sfx:
                    cmd += ["-i", sfx_path]
                cmd += ["-filter_complex", filter_complex,
                        "-map", vmap, "-map", "[a]"]
            else:
                cmd = ["ffmpeg", "-y",
                       "-ss", str(part_start), "-t", str(part_duration), "-i", video_path]

            cmd += [*video_encode_args(23),
                    "-pix_fmt", "yuv420p", "-r", "30",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                    "-fps_mode", "cfr",
                    "-muxpreload", "0", "-muxdelay", "0",
                    "-f", "mpegts", seg_path]

            if part_type == "bridge":
                print(f"    [{i+1}/{n}] {part_start:.1f}s ~ {part_end:.1f}s 브리지(클로즈업) 중...", flush=True)
            elif part_type == "punchin_bridge":
                print(f"    [{i+1}/{n}] {part_start:.1f}s ~ {part_end:.1f}s 펀치인(캠크롭) 중...", flush=True)
            else:
                print(f"    [{i+1}/{n}] {part_start:.1f}s ~ {part_end:.1f}s 컷 중...", flush=True)
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


def parse_punchin_times(times_str: str) -> List[float]:
    """시간 문자열을 초 리스트로 파싱.

    형식: "12:30, 45:02, 1:03:11" 등 쉼표로 구분.
    각 항목은 h:mm:ss / mm:ss / ss 지원. 잘못된 항목은 로그 후 무시.
    반환: 오름차순 정렬된 초 리스트.
    """
    if not times_str or not times_str.strip():
        return []

    result = []
    for item in times_str.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parts = item.split(":")
            parts_float = [float(p) for p in parts]
            if len(parts_float) == 1:
                t = parts_float[0]
            elif len(parts_float) == 2:
                t = parts_float[0] * 60 + parts_float[1]
            elif len(parts_float) == 3:
                t = parts_float[0] * 3600 + parts_float[1] * 60 + parts_float[2]
            else:
                print(f"  (펀치인 시간 형식 오류 무시: {item})")
                continue
            result.append(t)
        except (ValueError, IndexError) as e:
            print(f"  (펀치인 시간 파싱 오류 무시: {item} - {e})")
            continue

    result.sort()
    return result


def map_punchin_times(times: List[float], segments: List[Tuple[float, float]]) -> dict:
    """각 시간을 포함하는 구간에 매핑.

    반환: {구간인덱스: [시간1, 시간2, ...]}
    어느 구간에도 포함되지 않으면 로그 후 제외.
    """
    result = {}
    for time_sec in times:
        found = False
        for i, (start, end) in enumerate(segments):
            if start <= time_sec < end:
                if i not in result:
                    result[i] = []
                result[i].append(time_sec)
                found = True
                break
        if not found:
            print(f"  펀치인 {time_sec:.1f}s - 하이라이트 구간에 포함되지 않아 생략")

    return result


def compute_punchin_times(video_path: str, segments: List[Tuple[float, float]], level: str,
                          tmpdir: str, closeup_sec: float = 1.5) -> dict:
    """구간 중간의 오디오 에너지 피크에서 펀치인 시점을 고른다.

    level: "low"(구간 피크 상위 1/3만) / "mid"(상위 2/3) / "high"(모든 구간).
    반환: {구간인덱스: 원본기준 펀치인 시작초}

    각 구간에서 [start+2, end-closeup_sec-4] 범위 내 에너지 최대 윈도 시점을 피크로.
    범위가 음수면 그 구간은 제외 (브리지와 겹침 방지).
    """
    if level == "none":
        return {}

    # 전체 오디오를 16kHz mono로 추출
    wav_path = os.path.join(tmpdir, "punchin_audio.wav")
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-q:a", "9",
            wav_path
        ]
        run_ffmpeg(cmd, label="(펀치인 오디오 추출)")
    except Exception as e:
        print(f"  (펀치인 오디오 추출 실패: {e})")
        return {}

    # compute_energy로 에너지 계산
    try:
        energy, window_sec = compute_energy(wav_path, window_sec=0.5)
    except Exception as e:
        print(f"  (에너지 계산 실패: {e})")
        return {}

    # 각 구간에서 피크 찾기
    punchin_times = {}
    peaks_with_energy = []

    for i, (start, end) in enumerate(segments):
        # 범위: [start+2, end-closeup_sec-4]
        search_start = start + 2.0
        search_end = end - closeup_sec - 4.0

        if search_end <= search_start:
            # 범위가 음수 또는 매우 작음 -> 제외
            continue

        # 해당 범위 내 에너지 윈도 찾기
        start_win = int(search_start / window_sec)
        end_win = int(search_end / window_sec)

        if start_win >= len(energy) or end_win <= start_win:
            continue

        # 범위 내 최대 에너지 윈도
        range_energy = energy[start_win:end_win]
        if len(range_energy) == 0:
            continue

        max_idx = np.argmax(range_energy)
        peak_time = start + (search_start - start) + (max_idx * window_sec)
        peak_energy = range_energy[max_idx]

        peaks_with_energy.append((i, peak_time, peak_energy))

    # level에 따라 상위 구간만 선택
    if not peaks_with_energy:
        return {}

    # 에너지 기준으로 정렬
    peaks_with_energy.sort(key=lambda x: x[2], reverse=True)

    if level == "low":
        ratio = 1 / 3
    elif level == "mid":
        ratio = 2 / 3
    else:  # "high"
        ratio = 1.0

    cutoff_idx = max(1, int(len(peaks_with_energy) * ratio))
    selected_peaks = peaks_with_energy[:cutoff_idx]

    # 구간 인덱스로 정렬, 리스트 형식으로 저장
    for seg_idx, peak_time, _ in selected_peaks:
        if seg_idx not in punchin_times:
            punchin_times[seg_idx] = []
        punchin_times[seg_idx].append(peak_time)

    # 로그 출력
    if punchin_times:
        all_times = []
        for times_list in punchin_times.values():
            all_times.extend(times_list)
        times_str = ", ".join(f"{t:.1f}s" for t in sorted(all_times))
        total_count = len(all_times)
        print(f"  펀치인 {total_count}곳: {times_str}")

    return punchin_times



def silence_cut(video_path: str, out_path: str, tmpdir: str, threshold_db: float = -30.0,
                min_silence: float = 0.4, keep_pad: float = 0.06) -> bool:
    """Detect silent gaps and remove them (jump cut) to tighten pacing.

    Algorithm:
    1. Run ffmpeg silencedetect to find silence_start/silence_end events
    2. Invert silence list to get "keep" (non-silent) segments
    3. Add keep_pad to segment boundaries (soften cuts), clamp to [0, duration]
    4. Merge segments closer than ~0.15s and drop segments < ~0.30s
    5. If nothing to cut (kept ≈ full duration, or 0/1 segments), return False (no-op)
    6. Otherwise cut_and_concat() the kept segments and return True

    Returns: True if silence was cut, False if no significant silence found (original unchanged).
    Prints summary: original duration → kept duration, % removed.
    Never crashes: wrap detection in try/except, fall back to False on any error.
    """
    try:
        # Get video duration
        dur = get_duration(video_path)

        # Run silencedetect filter to find silence boundaries
        cmd = [
            "ffmpeg", "-i", os.path.abspath(video_path),
            "-af", f"silencedetect=noise={threshold_db}dB:d={min_silence}",
            "-f", "null", "-"
        ]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            **_PROC_KW
        )
        _, stderr = proc.communicate()

        if proc.returncode != 0:
            return False

        # Parse silence_start and silence_end lines from ffmpeg stderr
        # Example: "silence_start: 1.234" and "silence_end: 2.345"
        silence_intervals = []
        lines = (stderr or "").split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            if "silence_start:" in line:
                try:
                    start_str = line.split("silence_start:")[-1].strip().split()[0]
                    start = float(start_str)
                    # Look for corresponding silence_end in next few lines
                    end = None
                    for j in range(i+1, min(i+5, len(lines))):
                        if "silence_end:" in lines[j]:
                            # silence_end 줄은 '2.345 | silence_duration: 1.1' 형식이라
                            # 첫 토큰(숫자)만 취해야 float 파싱이 된다.
                            end_str = lines[j].split("silence_end:")[-1].strip().split()[0]
                            end = float(end_str)
                            break
                    if end is not None:
                        silence_intervals.append((max(0.0, start), min(dur, end)))
                        i = j
                    else:
                        i += 1
                except (ValueError, IndexError):
                    i += 1
            else:
                i += 1

        # Build "keep" segments = inverse of silence intervals
        keep_segments = []
        last_end = 0.0
        for sil_start, sil_end in sorted(silence_intervals):
            if sil_start > last_end:
                keep_segments.append((last_end, sil_start))
            last_end = max(last_end, sil_end)
        if last_end < dur:
            keep_segments.append((last_end, dur))

        # Apply keep_pad softening and clamp
        padded = []
        for seg_start, seg_end in keep_segments:
            new_start = max(0.0, seg_start - keep_pad)
            new_end = min(dur, seg_end + keep_pad)
            padded.append((new_start, new_end))

        # Merge nearby segments (within 0.15s) and drop very short ones (< 0.30s)
        merged = []
        for start, end in sorted(padded):
            if end - start < 0.30:
                continue
            if merged and start - merged[-1][1] < 0.15:
                # Merge with previous
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        # Check if there's actually silence to remove
        total_kept = sum(e - s for s, e in merged)

        # If we kept almost everything or have 0/1 segments, skip cutting
        if not merged or len(merged) <= 1 or abs(total_kept - dur) < 0.5:
            return False

        # Cut and concatenate
        cut_and_concat(video_path, merged, out_path, tmpdir,
                       transition_style="none", sfx_kind="none")

        # Print summary
        pct = (1.0 - total_kept / dur) * 100.0 if dur > 0 else 0.0
        print(f"  (무음 구간 제거: {dur:.1f}s → {total_kept:.1f}s, -{pct:.1f}%)")

        return True

    except Exception as e:
        # Fall back gracefully on any error
        return False


def safe_filename(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", title)[:80]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
                        help="화면 전환 스타일: none(없음) / black(암전) / white(화이트 플래시) / closeup(클로즈업). 기본 black")
    parser.add_argument("--cam-region", default="br",
                        help="캠 위치 (closeup/punchin 모드): tl(좌상) tr(우상) bl(좌하) br(우하) 또는 x,y,w,h 픽셀 문자열. 기본 br(우하단)")
    parser.add_argument("--closeup-every", type=int, default=1,
                        help="클로즈업을 몇 번의 전환마다 넣을지 (1=매번, 2=2회당 1회, 3=3회당 1회). 기본 1")
    parser.add_argument("--closeup-sec", type=float, default=1.5,
                        help="클로즈업/펀치인 길이(초). 기본 1.5")
    parser.add_argument("--sfx", dest="sfx_kind", default="whoosh",
                        choices=list(SFX_SPECS.keys()),
                        help="전환 효과음: none / whoosh(휙) / swoosh(스와이프) / "
                             "beep(삑) / pop(팝) / impact(임팩트). 기본 whoosh")
    parser.add_argument("--punchin", default="none",
                        choices=["none", "low", "mid", "high"],
                        help="구간 중간 캠 강조(펀치인): none(끔) / low(적게) / mid(보통) / high(많이). 기본 none")
    parser.add_argument("--punchin-times", default="",
                        help="펀치인 시간 직접 지정 (선택). 예: 12:30, 45:02, 1:03:11 (쉼표 구분, 원본 영상 기준). "
                             "자동(--punchin level)과 병합됨 - 같은 구간에서 3초 이내 중복은 제거")
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
    parser.add_argument("--jump-cut", action="store_true",
                        help="무음 구간 자동 컷(점프컷)으로 템포를 높입니다")
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
        set_hw_encoding(False)

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
            video_path, title = download_video(args.url, tmpdir, max_height=args.max_height, cookies_browser=args.cookies_browser)
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

        # 목소리 대역 에너지 (항상 계산)
        print(f"  목소리 대역(200-3800Hz) 에너지 계산 중...")
        voice_energy, _ = compute_voice_energy(wav_path, tmpdir, window_sec=0.5)

        # 채팅 반응 분석 (선택사항)
        chat_curve = None
        chat_sample_sec = None
        chat_weight = None
        if args.chat_analysis:
            cam_region = None
            # cam_region auto 아닌 경우에도 채팅 제외용으로 detect_cam_region 시도
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

        print(f"[5/6] Finding exciting segments (target: {args.target_min} min)...")
        target_sec = int(args.target_min * 60)
        segments = find_exciting_segments(
            energy, window_sec, whisper_result,
            target_sec=target_sec,
            expand_before=args.expand_before,
            expand_after=args.expand_after,
            bridge_gap=args.bridge_gap,
            chat_curve=chat_curve,
            chat_sample_sec=chat_sample_sec,
            chat_weight=chat_weight,
            voice_energy=voice_energy,
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

        # Build raw summary first (without jump-cut)
        summary_raw = os.path.join(tmpdir, "summary_raw.mp4")

        # compute_punchin_times if needed + merge with manual times
        punchins = {}
        manual_punchins = {}

        # 자동 펀치인
        if args.punchin != "none":
            punchins = compute_punchin_times(video_path, segments, args.punchin, tmpdir,
                                             closeup_sec=args.closeup_sec)

        # 수동 펀치인 파싱 및 매핑
        if args.punchin_times.strip():
            manual_times = parse_punchin_times(args.punchin_times)
            manual_punchins = map_punchin_times(manual_times, segments)

        # 병합: 수동 시간과 자동 시간이 3초 이내로 겹치면 자동 제거
        if manual_punchins and punchins:
            for seg_idx in manual_punchins:
                if seg_idx in punchins:
                    # 같은 구간에 수동과 자동이 모두 있을 때
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

        cut_and_concat(video_path, segments, summary_raw, tmpdir,
                       transition_style=args.transition_style, sfx_kind=args.sfx_kind,
                       cam_region=args.cam_region, closeup_sec=args.closeup_sec,
                       closeup_every=args.closeup_every, punchins=punchins)

        # Apply jump-cut if requested
        summary_video = summary_raw
        srt_segments = segments  # segments list for SRT (may change if jump-cut happens)
        if args.jump_cut:
            summary_cut = os.path.join(tmpdir, "summary_cut.mp4")
            cut_happened = silence_cut(summary_raw, summary_cut, tmpdir)
            if cut_happened:
                summary_video = summary_cut
                # Re-generate SRT if subtitles are on (jump-cut changes timing)
                wav_path_cut = os.path.join(tmpdir, "audio_cut.wav")
                extract_audio(summary_video, wav_path_cut)
                whisper_result = transcribe(wav_path_cut, args.model, args.lang, args.prompt)
                # For SRT, use full video duration since segments no longer align
                srt_segments = [(0.0, get_duration(summary_video))]

        print(f"[7/7] Building subtitles...")
        srt_content = build_srt(whisper_result, srt_segments)
        with open(out_srt, "w", encoding="utf-8") as f:
            f.write(srt_content)

        # Copy final summary video to output
        shutil.copy2(summary_video, out_video)

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

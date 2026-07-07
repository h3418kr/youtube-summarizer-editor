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


def download_video(url: str, tmpdir: str, max_height: int = 720) -> Tuple[str, str]:
    info_raw = run(YTDLP + ["--print", "%(id)s|||%(title)s", "--no-playlist", url])
    vid_id, title = info_raw.split("|||", 1)
    out_path = os.path.join(tmpdir, f"{vid_id}.mp4")
    fmt = (
        f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]"
        f"/bestvideo[height<={max_height}]+bestaudio"
        f"/best[height<={max_height}]/best"
    )
    subprocess.run(
        YTDLP + ["-f", fmt, "--merge-output-format", "mp4",
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
                   cam_region: str = "br", closeup_sec: float = 1.5) -> None:
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

    cam_region: "br"(기본) 등 프리셋 또는 "x,y,w,h" 픽셀 문자열 (closeup일 때만 사용)
    closeup_sec: closeup 브리지 길이 (기본 1.5초)
    """
    has_video_fx = transition_style in ("black", "white", "closeup")
    sfx_path, sfx_len = make_sfx(tmpdir, sfx_kind)

    # closeup 브리지를 위한 해상도 취득
    W, H = None, None
    if transition_style == "closeup":
        try:
            W, H = get_media_size(video_path)
        except Exception:
            print(f"  (해상도 취득 실패, closeup 비활성화)")
            transition_style = "none"
            has_video_fx = False

    # "auto" 감지
    if cam_region == "auto":
        detected = detect_cam_region(video_path, tmpdir)
        if detected:
            cam_region = ",".join(str(v) for v in detected)
        else:
            print("  (자동 감지 실패 - 우하단 프리셋 사용)")
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
        if transition_style == "closeup" and W and H:
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

        # closeup 브리지 로직: 구간을 part1/part2로 분할
        is_last = (i == n - 1)
        parts_to_encode = []
        if transition_style == "closeup" and not is_last and duration >= closeup_sec * 3:
            # part1: start ~ (end - closeup_sec)
            part1_start, part1_end = start, end - closeup_sec
            # part2: (end - closeup_sec) ~ end (브리지)
            part2_start, part2_end = end - closeup_sec, end
            parts_to_encode = [(part1_start, part1_end, False), (part2_start, part2_end, True)]
        else:
            # closeup 아니거나 마지막이거나 너무 짧으면 일반 인코딩
            parts_to_encode = [(start, end, False)]

        for part_idx, (part_start, part_end, is_closeup_bridge) in enumerate(parts_to_encode):
            if is_closeup_bridge:
                seg_path = os.path.join(tmpdir, f"seg_{i:04d}_bridge.ts")
            else:
                seg_path = os.path.join(tmpdir, f"seg_{i:04d}.ts") if part_idx == 0 else os.path.join(tmpdir, f"seg_{i:04d}_part1.ts")

            part_duration = part_end - part_start
            add_sfx = bool(sfx_path) and i < n - 1 and is_closeup_bridge  # 브리지 끝에만 효과음
            do_fx = (has_video_fx or add_sfx) and part_duration > 1.0

            if do_fx:
                f = min(fade, part_duration / 4)    # 매우 짧은 클립 보호
                af = min(f, 0.15)                   # 오디오 페이드 인

                # ── 비디오 브랜치 ──
                if is_closeup_bridge and transition_style == "closeup":
                    # 클로즈업 브리지: 캠 영역 crop + 확대
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

            if is_closeup_bridge:
                print(f"    [{i+1}/{n}] {part_start:.1f}s ~ {part_end:.1f}s 브리지(클로즈업) 중...", flush=True)
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
                        help="캠 위치 (closeup 모드): tl(좌상) tr(우상) bl(좌하) br(우하) 또는 x,y,w,h 픽셀 문자열. 기본 br(우하단)")
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
    parser.add_argument("--jump-cut", action="store_true",
                        help="무음 구간 자동 컷(점프컷)으로 템포를 높입니다")
    parser.add_argument("--cpu-encode", action="store_true",
                        help="GPU 가속 인코딩 끄기 (호환성 문제 시)")
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

        # Build raw summary first (without jump-cut)
        summary_raw = os.path.join(tmpdir, "summary_raw.mp4")
        cut_and_concat(video_path, segments, summary_raw, tmpdir,
                       transition_style=args.transition_style, sfx_kind=args.sfx_kind,
                       cam_region=args.cam_region, closeup_sec=1.5)

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

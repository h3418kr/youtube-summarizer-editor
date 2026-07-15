"""완성 영상 만들기 - 영상 + 자막(SRT) + 썸네일 + 인트로/아웃트로 영상을
하나의 완성 mp4로 합친다.

  - 자막: 영상 화면에 새겨넣기(하드섭). 어디서 재생해도 자막이 보인다.
  - 썸네일: (1) 영상 맨 앞에 2~3초 인트로로 붙이고, (2) mp4 표지(커버)로도 삽입.
  - 인트로/아웃트로 영상: 본편 앞/뒤에 별도의 영상 클립을 붙인다.
    해상도/비율이 달라도 본편 규격에 맞춰 자동 변환해 이어붙인다.
  - 배경음악(BGM): 완성 영상 전체 길이에 맞춰 반복/컷 하여 기존 오디오와 섞는다.

ffmpeg 만 사용하며, 배포 폴더 옆의 ffmpeg/bin 을 자동으로 PATH 에 추가한다.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np


def _setup_bundled_paths():
    """포터블 배포용: 스크립트 폴더 옆의 ffmpeg/bin 을 PATH 에 추가."""
    base = os.path.dirname(os.path.abspath(__file__))
    for rel in (os.path.join("ffmpeg", "bin"), "ffmpeg"):
        p = os.path.join(base, rel)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "ffmpeg.exe")):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            break


_setup_bundled_paths()

# summarizer 모듈의 GPU 인코딩 헬퍼 import
# (포터블 임베디드 파이썬은 현재 스크립트 폴더를 sys.path에 자동 추가하지 않으므로 명시)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarizer import video_encode_args, extract_audio, compute_energy, compute_voice_energy

# ── Font helper (bundled fonts) ──────────────────────────────────────────────
def bundled_fonts_dir():
    """번들 폰트 폴더를 찾는다."""
    base = os.path.dirname(os.path.abspath(__file__))
    dist_fonts = os.path.join(base, "배포_요약기_무설치", "fonts")
    if os.path.isdir(dist_fonts):
        return dist_fonts
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
_PROC_KW = dict(stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW)


def run(cmd, **kwargs):
    kw = dict(_PROC_KW)
    kw.update(kwargs)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", **kw)
    return result.stdout.strip()


def run_ffmpeg(cmd, label: str = "", cwd: str = None) -> None:
    """ffmpeg 실행: 콘솔창 숨김 + stdin 차단 + 진행상황(time=) 스트리밍."""
    if cmd and "ffmpeg" in os.path.basename(str(cmd[0])).lower():
        cmd = [cmd[0], "-nostdin", "-hide_banner"] + list(cmd[1:])

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", cwd=cwd, **_PROC_KW,
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


def get_video_props(path: str):
    """영상의 (가로, 세로, 프레임레이트문자열) 을 ffprobe 로 읽는다."""
    out = run(["ffprobe", "-v", "quiet", "-print_format", "json",
               "-select_streams", "v:0",
               "-show_entries", "stream=width,height,r_frame_rate", path])
    data = json.loads(out)
    st = data["streams"][0]
    w = int(st["width"])
    h = int(st["height"])
    fps = st.get("r_frame_rate", "30/1")
    if not fps or fps == "0/0":
        fps = "30/1"
    return w, h, fps


# 인코딩 공통 옵션 (인트로/본편이 같은 규격이라야 재인코딩 없이 이어붙일 수 있다)
def _enc_opts(fps: str):
    return [*video_encode_args(23),
            "-pix_fmt", "yuv420p", "-r", fps,
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-fps_mode", "cfr", "-muxpreload", "0", "-muxdelay", "0",
            "-f", "mpegts"]


def make_intro(thumb: str, w: int, h: int, fps: str, seconds: float,
               out_ts: str) -> None:
    """썸네일을 영상 규격(WxH, fps)에 맞춰 seconds 초짜리 무음 인트로로 만든다."""
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p")
    cmd = ["ffmpeg", "-y",
           "-loop", "1", "-t", f"{seconds}", "-i", thumb,
           "-f", "lavfi", "-t", f"{seconds}", "-i", "anullsrc=r=44100:cl=stereo",
           "-vf", vf] + _enc_opts(fps) + ["-shortest", out_ts]
    run_ffmpeg(cmd, label="(인트로)")


def has_audio(path: str) -> bool:
    """영상에 오디오 스트림이 있는지 ffprobe 로 확인."""
    try:
        out = run(["ffprobe", "-v", "quiet", "-select_streams", "a:0",
                   "-show_entries", "stream=index", "-of", "csv=p=0", path])
        return bool(out.strip())
    except Exception:
        return False


def prep_clip(clip: str, w: int, h: int, fps: str, out_ts: str,
              label: str = "(클립)") -> None:
    """인트로/아웃트로 영상을 본편 규격(WxH·fps·44.1k 스테레오)에 맞춰 TS 로 변환.

    해상도/비율이 달라도 레터박스(pad)로 맞추고, 오디오가 없으면 무음을 넣어
    본편과 동일한 스트림 구성을 만들어야 재인코딩 없이 이어붙일 수 있다.
    """
    clip = os.path.abspath(clip)
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p")
    if has_audio(clip):
        cmd = ["ffmpeg", "-y", "-i", clip,
               "-filter_complex", f"[0:v]{vf}[v]",
               "-map", "[v]", "-map", "0:a:0"] + _enc_opts(fps) + [out_ts]
    else:
        cmd = ["ffmpeg", "-y", "-i", clip,
               "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
               "-filter_complex", f"[0:v]{vf}[v]",
               "-map", "[v]", "-map", "1:a", "-shortest"] + _enc_opts(fps) + [out_ts]
    run_ffmpeg(cmd, label=label)


def _loudnorm(stage: str, out_path: str) -> None:
    """유튜브 표준(-14 LUFS)으로 음량을 2패스 정규화한다.

    1패스에서 입력 라우드니스를 측정(print_format=json)하고, 그 값을 2패스에
    넣어 linear 정규화하면 목표(-14 LUFS)에 정확히 맞는다. 단일 패스는 ±1~2 LUFS
    오차가 나서 이미 -14 근처인 영상이 오히려 멀어지기도 한다.
    측정/파싱 실패 시 단일 패스로 폴백한다. 비디오는 스트림 카피(빠름).
    """
    target = "I=-14:TP=-1.5:LRA=11"
    measured = None
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostdin", "-i", stage,
             "-af", f"loudnorm={target}:print_format=json", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            **_PROC_KW)
        m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.DOTALL)
        if m:
            measured = json.loads(m.group(0))
    except Exception:
        measured = None

    if measured:
        af = (f"loudnorm={target}:"
              f"measured_I={measured['input_i']}:"
              f"measured_TP={measured['input_tp']}:"
              f"measured_LRA={measured['input_lra']}:"
              f"measured_thresh={measured['input_thresh']}:"
              f"offset={measured['target_offset']}:linear=true")
    else:
        af = f"loudnorm={target}"  # 측정 실패 시 단일 패스
    run_ffmpeg(
        ["ffmpeg", "-i", stage, "-c:v", "copy", "-af", af,
         "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
         "-movflags", "+faststart", out_path],
        label="(음량 정규화)")


WM_POSITIONS = {"tl": "좌상단", "tr": "우상단", "bl": "좌하단", "br": "우하단"}

# ── AI 자동 키워드(Gemini) ─────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"
# 모델이 404(지원종료)/429(할당량)면 자동으로 아래 모델로 대체 시도한다.
GEMINI_FALLBACK = "gemini-flash-latest"


def _media_wh(path: str):
    out = run(["ffprobe", "-v", "quiet", "-print_format", "json",
               "-select_streams", "v:0", "-show_streams", path])
    st = json.loads(out)["streams"][0]
    return int(st["width"]), int(st["height"])


def _parse_time_tok(tok: str) -> float:
    """'83' / '1:23' / '00:01:23' / '1:23,500' -> 초(float)."""
    tok = tok.strip().replace(",", ".")
    parts = [float(p) for p in tok.split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


_TIME_RANGE = re.compile(r'(\d[\d:.,]*)\s*[-~]\s*(\d[\d:.,]*)')


def parse_srt(srt_path: str):
    """SRT 파일을 파싱해 [(start_sec, end_sec, text), ...] 반환.

    본편 타임라인 기준 시간(초). 본편이 인트로/아웃트로를 포함하면
    finalize 호출 시 adjust_srt_timing으로 조정해야 한다.
    """
    cues = []
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return cues

    # SRT 형식: idx \n HH:MM:SS,mmm --> HH:MM:SS,mmm \n text \n \n
    pattern = re.compile(
        r'(\d+)\s*\n'
        r'(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*'
        r'(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*\n'
        r'(.*?)(?=\n\s*\n|\Z)',
        re.DOTALL
    )

    for m in pattern.finditer(text):
        try:
            h1, m1, s1, ms1 = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
            h2, m2, s2, ms2 = int(m.group(6)), int(m.group(7)), int(m.group(8)), int(m.group(9))
            start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
            end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
            txt = m.group(10).strip()
            if end > start:
                cues.append((start, end, txt))
        except Exception:
            pass

    return cues


def _parse_label_lines(text: str):
    """'시작-끝|키워드' 형식의 여러 줄을 (start, end, keyword) 리스트로.

    시간은 초/분:초/시:분:초 아무 형식이나 허용하고, 키워드 구분자는 '|'.
    (분:초의 콜론과 헷갈리지 않도록 구분자는 '|' 만 인정)
    """
    labels = []
    for line in text.splitlines():
        line = line.strip().strip("`").strip()
        if "|" not in line:
            continue
        tpart, kw = line.split("|", 1)
        kw = kw.strip().strip('"').strip("'").strip()
        if not kw:
            continue
        m = _TIME_RANGE.search(tpart)
        if not m:
            continue
        try:
            s = _parse_time_tok(m.group(1))
            e = _parse_time_tok(m.group(2))
        except Exception:
            continue
        if e > s:
            labels.append((s, e, kw))
    return labels


def gemini_labels(srt_path: str, api_key: str, model: str = GEMINI_MODEL):
    """SRT 자막을 Gemini 에 보내 구간별 핵심 키워드 라벨을 받아온다.

    반환: [(start_sec, end_sec, keyword), ...] (본편 타임라인 기준). 실패 시 [].
    """
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            srt_text = f.read()
    except Exception:
        return []
    if not srt_text.strip():
        return []
    prompt = (
        "다음은 게임 방송 하이라이트 영상의 자막(SRT)입니다. "
        "영상을 내용이 비슷한 3~8개 구간으로 나누고, 각 구간을 한눈에 나타내는 "
        "아주 짧은 한국어 키워드(2~8자) 하나씩을 붙이세요. "
        "설명 없이 각 줄을 '시작초-끝초|키워드' 형식으로만 출력하세요. "
        "시간은 자막의 초 단위 정수입니다.\n\n자막:\n" + srt_text
    )
    _NONE = "BLOCK_NONE"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        # 게임 방송 자막(전투·욕설 등)이 안전 필터에 걸려 빈 응답이 오는 것을 방지
        "safetySettings": [
            {"category": c, "threshold": _NONE} for c in (
                "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
        ],
    }
    body = json.dumps(payload).encode("utf-8")

    # 지정 모델 -> (404/429면) 대체 모델 순으로 시도
    candidates = [model] + ([GEMINI_FALLBACK] if model != GEMINI_FALLBACK else [])
    last_err = "알 수 없는 오류"
    for mdl in candidates:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{mdl}:generateContent?key={api_key}")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:200]
            except Exception:
                pass
            last_err = f"HTTP {e.code} ({mdl})"
            if e.code in (404, 429):   # 모델 미지원/할당량 -> 다음 후보 시도
                print(f"  ({mdl}: {e.code} - 다른 모델로 재시도)", flush=True)
                continue
            print(f"  (Gemini 요청 실패 {last_err}: {detail})", flush=True)
            return []
        except Exception as e:
            print(f"  (Gemini 연결 실패: {e})", flush=True)
            return []
        try:
            data = json.loads(raw)
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            last_err = f"응답 형식 이상 ({mdl})"
            continue
        labels = _parse_label_lines(text)
        if labels:
            print(f"  Gemini 키워드 {len(labels)}개 생성 (모델 {mdl})", flush=True)
            return labels
        last_err = "응답에서 키워드 파싱 실패: " + " / ".join(text.splitlines())[:150]
    print(f"  (Gemini 키워드 생성 실패: {last_err})", flush=True)
    return []


def _ass_ts(sec: float) -> str:
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


def _build_label_ass(w, h, labels, anchor, x, y, font, size):
    """키워드 라벨을 마크 아래(픽셀 좌표)에 배치한 ASS 문서."""
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\n"
        "ScaledBorderAndShadow: yes\nWrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: L,{font},{size},&H00FFFFFF,&HC8000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,3,1,{anchor},10,10,10,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )
    lines = []
    for s, e, text in labels:
        t = (str(text).replace("\\", "").replace("{", "(").replace("}", ")")
             .replace("\r", "").replace("\n", r"\N"))
        lines.append(f"Dialogue: 0,{_ass_ts(s)},{_ass_ts(e)},L,,0,0,0,,"
                     f"{{\\an{anchor}\\pos({x},{y})}}{t}")
    return head + "\n".join(lines) + "\n"


def _build_sub_ass(w, h, cues, impact_cues_set, font, size,
                    impact_size, impact_color, impact_pos, impact_pop):
    """SRT cues를 ASS로 변환. 임팩트 cue는 Impact 스타일.

    Args:
        cues: [(start_sec, end_sec, text), ...]
        impact_cues_set: set of (start_sec, end_sec, text) 임팩트 선정된 것들
        impact_color: "&H0000FFFF" (노랑) 형식, None이면 스킵
    """
    # ASS 헤더
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\n"
        "ScaledBorderAndShadow: yes\nWrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
    )

    # 기본 스타일
    head += (f"Style: Default,{font},{size},&H00FFFFFF,&H90000000,&H00000000,"
             f"0,0,0,0,100,100,0,0,1,2,1,2,10,10,28,1\n")

    # Impact 스타일 (임팩트 색상 지정된 경우만)
    if impact_color:
        # 위치별 Alignment (ASS: 1-9 키패드)
        # center: 5, top: 8, bottom: 2
        align_map = {"center": "5", "top": "8", "bottom": "2"}
        alignment = align_map.get(impact_pos, "5")
        margin_v = {"center": "0", "top": "180", "bottom": "220"}[impact_pos]
        head += (f"Style: Impact,{font},{impact_size},{impact_color},&H00000000,&H00000000,"
                 f"1,0,0,0,100,100,0,0,1,5,2,{alignment},10,10,{margin_v},1\n")

    head += "\n[Events]\n"
    head += "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

    # 각 cue별 이벤트 생성
    events = []
    for start, end, text in cues:
        # 텍스트 이스케이프
        t = (str(text).replace("\\", "").replace("{", "(").replace("}", ")")
             .replace("\r", "").replace("\n", r"\N"))

        is_impact = (start, end, text) in impact_cues_set
        if is_impact and impact_color:
            # Impact 스타일 + 팝 효과
            if impact_pop:
                # 팝 효과: 120ms에 걸쳐 40% -> 100%로 확대
                text_with_effect = f"{{\\fscx40\\fscy40\\t(0,120,\\fscx100\\fscy100)}}{t}"
            else:
                text_with_effect = t
            style = "Impact"
        else:
            # 기본 스타일 (임팩트가 아니거나 impact_color가 없으면 Default)
            style = "Default"
            text_with_effect = t

        events.append(f"Dialogue: 0,{_ass_ts(start)},{_ass_ts(end)},{style},,0,0,0,,{text_with_effect}")

    return head + "\n".join(events) + "\n"


def render_main(video: str, srt_name: str, w: int, h: int, fps: str,
                out_ts: str, cwd: str, font: str, font_size: int,
                burn_sub: bool, watermark: str = "", wm_pos: str = "tr",
                wm_scale: float = 0.12, wm_margin: int = 24,
                wm_colorkey: str = "", labels=None,
                label_font: str = "Paperlogy", label_size: int = 44,
                impact_cues=None, impact_size: int = 64,
                impact_color: str = "", impact_pos: str = "center",
                impact_pop: bool = True) -> None:
    """본편을 규격 통일 + (선택)자막 하드섭 + (선택)채널 마크 오버레이 하여 TS 로.

    마크는 본편에만 들어간다(인트로/아웃트로 TS 는 손대지 않으므로 자동으로
    본영상에만 남는다). subtitles 필터 경로는 Windows 이스케이프가 까다로워
    SRT 를 작업 폴더(cwd)에 복사한 뒤 상대 경로로 참조한다.

    wm_colorkey 를 주면(예: black/white/0xRRGGBB) 마크 이미지에서 그 배경색을
    투명 처리(colorkey)한 뒤 얹는다. 배경이 단색인 로고를 투명 없이 써도 된다.

    impact_cues: 임팩트로 선정된 cues [(start, end, text), ...]. None이면 무시.
    """
    labels = [(s, e, t) for (s, e, t) in (labels or []) if str(t).strip()]
    m = int(wm_margin)
    left = wm_pos in ("tl", "bl")
    top = wm_pos in ("tl", "tr")
    has_wm = bool(watermark) and os.path.isfile(watermark)

    inputs = ["-i", os.path.abspath(video)]
    fc_parts = []

    # 1) base: 규격 통일 + (선택)말소리 자막 하드섭
    base = f"[0:v]scale={w}:{h},setsar=1"
    if burn_sub and srt_name:
        # SRT를 파싱해 ASS로 변환 (임팩트 처리 포함)
        srt_path = os.path.join(cwd, srt_name)
        cues = parse_srt(srt_path)
        impact_cues_set = set(impact_cues or [])

        # ASS 생성
        ass_content = _build_sub_ass(w, h, cues, impact_cues_set, font, font_size,
                                     impact_size, impact_color, impact_pos, impact_pop)
        ass_path = os.path.join(cwd, "subs.ass")
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

        base += f",subtitles=subs.ass:fontsdir=."
    fc_parts.append(base + "[base]")
    cur = "[base]"

    # 2) 채널 마크 오버레이
    logo_h = 0
    if has_wm:
        lw = max(16, int(w * float(wm_scale)))
        ox = f"{m}" if left else f"W-w-{m}"
        oy = f"{m}" if top else f"H-h-{m}"
        inputs += ["-i", os.path.abspath(watermark)]
        # 배경색 투명 처리: colorkey 는 알파를 만들므로 rgba 로 두고 처리한다.
        logo = f"[1:v]format=rgba,scale={lw}:-1"
        if wm_colorkey:
            logo += f",colorkey={wm_colorkey}:0.30:0.10"
        fc_parts.append(logo + "[wm]")
        fc_parts.append(f"{cur}[wm]overlay={ox}:{oy}[vmark]")
        cur = "[vmark]"
        try:
            iw, ih = _media_wh(watermark)
            logo_h = int(lw * ih / iw) if iw else int(lw * 0.5)
        except Exception:
            logo_h = int(lw * 0.5)

    # 3) AI 키워드 라벨: 마크 아래(마크 없으면 코너)에 픽셀 좌표로 배치
    if labels:
        gap = 12
        reserve = logo_h if has_wm else int(h * 0.11)
        if top:
            l_anchor = 7 if left else 9
            ly = m + reserve + gap
        else:
            l_anchor = 1 if left else 3
            ly = h - m - reserve - gap
        lx = m if left else (w - m)
        ass = _build_label_ass(w, h, labels, l_anchor, lx, ly,
                               label_font, label_size)
        with open(os.path.join(cwd, "labels.ass"), "w", encoding="utf-8") as f:
            f.write(ass)
        fc_parts.append(f"{cur}subtitles=labels.ass:fontsdir=.[vlab]")
        cur = "[vlab]"

    # 4) 마무리 포맷
    fc_parts.append(f"{cur}format=yuv420p[v]")
    fc = ";".join(fc_parts)

    cmd = (["ffmpeg", "-y"] + inputs +
           ["-filter_complex", fc, "-map", "[v]", "-map", "0:a?"] +
           _enc_opts(fps) + [out_ts])
    run_ffmpeg(cmd, label="(본편)", cwd=cwd)


def _find_energy_peaks(energy: np.ndarray, window_sec: float, duration: float,
                       n_peaks: int = 3, min_gap: float = 15.0,
                       exclude_start: float = 3.0, exclude_end: float = 3.0) -> list:
    """에너지 곡선에서 피크를 찾는다.

    반환: [(peak_time_sec, peak_index), ...] (peak_time_sec: 본편 기준 초)
    """
    if len(energy) == 0:
        return []

    # z-score 정규화
    mean = np.mean(energy)
    std = np.std(energy)
    if std < 1e-6:
        return []
    z_score = (energy - mean) / std

    # 제외 구간 마스크
    exclude_start_idx = int(exclude_start / window_sec)
    exclude_end_idx = int((duration - exclude_end) / window_sec)
    mask = np.ones(len(energy), dtype=bool)
    mask[:exclude_start_idx] = False
    mask[exclude_end_idx:] = False
    z_score[~mask] = -np.inf

    # 피크 찾기 (min_gap 유지)
    peaks = []
    min_gap_idx = int(min_gap / window_sec)
    while len(peaks) < n_peaks:
        if np.all(z_score <= -np.inf):
            break
        idx = np.argmax(z_score)
        if z_score[idx] <= -np.inf:
            break
        peaks.append(idx)
        # 주변 제외
        start = max(0, idx - min_gap_idx)
        end = min(len(z_score), idx + min_gap_idx + 1)
        z_score[start:end] = -np.inf

    # 시간으로 변환 (제시하는 시간은 피크 자체)
    result = [(idx * window_sec, idx) for idx in peaks]
    result.sort(key=lambda x: x[0])  # 시간 순서대로
    return result


def _extract_teaser_clips(video: str, w: int, h: int, fps: str, tmpdir: str,
                          n_clips: int = 3, clip_duration: float = 1.5) -> list:
    """본편에서 "가장 뜨거운 순간" N개를 추출.

    반환: [teaser_ts_file_1, teaser_ts_file_2, ...] (각 clip_duration초)

    Args:
        clip_duration: 각 티저 컷의 길이(초), 기본 1.5. 범위 0.5~5.0으로 클램프됨.
    """
    # clip_duration을 0.5~5.0 범위로 클램프
    clip_duration = max(0.5, min(5.0, clip_duration))
    # 1) 오디오 추출
    wav_path = os.path.join(tmpdir, "main_audio.wav")
    extract_audio(video, wav_path)

    # 2) 에너지 계산 (전체 에너지 + 목소리 에너지)
    energy_arr, window_sec = compute_energy(wav_path)
    voice_arr, _ = compute_voice_energy(wav_path, tmpdir)

    # 에너지 합산 (voice 가중치 높임)
    if len(voice_arr) > 0:
        # 길이 맞추기
        min_len = min(len(energy_arr), len(voice_arr))
        combined = energy_arr[:min_len] + voice_arr[:min_len] * 2.0
    else:
        combined = energy_arr

    # 영상 길이(초)
    video_dur = get_video_duration(video)

    # 3) 피크 찾기
    peaks = _find_energy_peaks(combined, window_sec, video_dur, n_peaks=n_clips,
                               min_gap=15.0, exclude_start=3.0, exclude_end=3.0)
    if not peaks:
        print(f"[티저] 가능한 피크를 찾을 수 없습니다.", flush=True)
        return []

    # 4) 각 피크에서 clip_duration 초짜리 클립 추출
    teaser_files = []
    for i, (peak_time, _) in enumerate(peaks, 1):
        clip_start = max(0.0, peak_time - clip_duration / 2)
        clip_end = min(video_dur, clip_start + clip_duration)
        # 길이 보정
        if clip_end - clip_start < clip_duration * 0.9:
            clip_start = max(0.0, clip_end - clip_duration)

        out_ts = os.path.join(tmpdir, f"teaser_cut_{i}.ts")
        cmd = ["ffmpeg", "-y", "-i", os.path.abspath(video),
               "-ss", f"{clip_start}", "-to", f"{clip_end}",
               "-vf", f"scale={w}:{h},setsar=1,format=yuv420p"] + _enc_opts(fps) + [out_ts]
        run_ffmpeg(cmd, label=f"(티저 컷 {i}/{len(peaks)})")
        teaser_files.append(out_ts)

        # 로그: 피크 시간 출력
        print(f"[티저] 컷 {i}: 원본 {peak_time:.2f}초 (±{clip_duration/2:.2f}초)", flush=True)

    return teaser_files


def select_impact_cues(cues, energy_arr, voice_arr, window_sec, level,
                       min_gap=8.0, max_len=6.0):
    """에너지 기반으로 임팩트 줄 선정.

    Args:
        cues: [(start_sec, end_sec, text), ...]. 본편 타임라인.
        energy_arr: 전체 에너지 배열 (compute_energy 결과)
        voice_arr: 음성 에너지 배열 (compute_voice_energy 결과)
        window_sec: 에너지 계산 윈도우 크기(초)
        level: "low"(5%) / "mid"(10%) / "high"(20%)
        min_gap: 선정된 것끼리 이만큼 미만 간격이면 점수 낮은 쪽 제외(초)
        max_len: 이보다 긴 cue는 제외(초)

    Returns:
        [(start_sec, end_sec, text), ...] 임팩트로 선정된 줄들 (시간순)
    """
    # 점수 계산: combined = 0.5*energy_z + 1.0*voice_z
    if len(energy_arr) == 0:
        return []

    # z-score 정규화
    energy_mean = np.mean(energy_arr)
    energy_std = np.std(energy_arr)
    if energy_std < 1e-6:
        energy_z = np.zeros_like(energy_arr)
    else:
        energy_z = (energy_arr - energy_mean) / energy_std

    voice_z = np.zeros_like(energy_arr)
    if len(voice_arr) > 0:
        voice_mean = np.mean(voice_arr)
        voice_std = np.std(voice_arr)
        min_len = min(len(energy_arr), len(voice_arr))
        if voice_std > 1e-6:
            voice_z[:min_len] = (voice_arr[:min_len] - voice_mean) / voice_std

    combined = 0.5 * energy_z + 1.0 * voice_z

    # 각 cue의 점수 = cue 구간 내 combined 최댓값
    cue_scores = []
    for start, end, text in cues:
        # 제외: 길이 초과 or 텍스트 40자 초과
        if end - start > max_len:
            continue
        if len(text.replace("\n", "").replace("\r", "")) > 40:
            continue

        # 구간 내 최고 점수
        start_idx = int(start / window_sec)
        end_idx = int(end / window_sec) + 1
        start_idx = max(0, min(start_idx, len(combined) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(combined)))

        max_score = float(np.max(combined[start_idx:end_idx])) if start_idx < end_idx else 0.0
        cue_scores.append((max_score, start, end, text))

    if not cue_scores:
        return []

    # 상위 N% 선정 (level에 따라)
    level_ratio = {"low": 0.05, "mid": 0.10, "high": 0.20}.get(level, 0.10)
    n_select = max(1, int(len(cue_scores) * level_ratio))
    sorted_cues = sorted(cue_scores, key=lambda x: x[0], reverse=True)[:n_select]
    sorted_cues = sorted(sorted_cues, key=lambda x: x[1])  # 시간 순서대로

    # min_gap 유지 (그리디: 점수 높은 순으로 채택, 기존 채택분과 가까우면 제외)
    result = []
    for score, start, end, text in sorted(sorted_cues, key=lambda x: x[0], reverse=True):
        too_close = any(
            min(abs(start - prev_end), abs(prev_start - end)) < min_gap
            for _, prev_start, prev_end, _ in result
        )
        if not too_close:
            result.append((score, start, end, text))
    result.sort(key=lambda x: x[1])  # 시간 순서대로

    # 로그: "임팩트 자막 N줄: 12.4s '아니 이게 왜 죽어', ..."
    print(f"  임팩트 자막 {len(result)}줄:", end="", flush=True)
    for _, start, end, text in result:
        text_short = text.replace("\n", " ")[:40]
        print(f" {start:.1f}s '{text_short}'", end="", flush=True)
    print("", flush=True)

    return [(s, e, t) for _, s, e, t in result]


def get_video_duration(video_path: str) -> float:
    """ffprobe로 영상 길이를 초(float) 단위로 구한다."""
    try:
        out = run(["ffprobe", "-v", "quiet", "-print_format", "json",
                   "-select_streams", "v:0", "-show_entries", "stream=duration",
                   video_path])
        data = json.loads(out)
        if data.get("streams"):
            dur = float(data["streams"][0].get("duration", 0))
            return dur
    except Exception:
        pass
    return 0.0


def concat_ts(ts_files, out_mp4: str, tmpdir: str) -> None:
    """TS 조각들을 concat 디먹서로 이어붙여 mp4 로 담는다(재인코딩 X)."""
    list_path = os.path.join(tmpdir, "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for ts in ts_files:
            safe = ts.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    run_ffmpeg(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
         "-c", "copy", "-bsf:a", "aac_adtstoasc",
         "-movflags", "+faststart", out_mp4],
        label="(이어붙이기)",
    )


def mix_bgm_into_ts(main_ts: str, bgm: str, out_ts: str, volume: float = 0.25) -> None:
    """본편 TS 조각에만 배경음악을 섞어 새 TS 로 만든다(영상은 재인코딩 X).

    BGM 을 본편 클립에만 넣으므로 인트로/아웃트로/썸네일 구간에는 깔리지 않는다.
    -stream_loop -1 로 BGM 을 무한 반복시키고, amix 의 duration=first 로 본편
    길이에 정확히 맞춘다(짧으면 반복 채움, 길면 잘라냄). normalize=0 으로 원본
    말소리 볼륨은 그대로 두고 BGM 만 낮추며, alimiter 로 합산 시 클리핑을 막는다.
    이어붙이기(concat)와 호환되도록 오디오 규격을 본편과 동일하게 맞춰 TS 로 낸다.
    """
    bgm = os.path.abspath(bgm)
    fc = (f"[1:a]volume={volume}[bg];"
          f"[0:a][bg]amix=inputs=2:duration=first:normalize=0[mix];"
          f"[mix]alimiter=limit=0.95[a]")
    run_ffmpeg(
        ["ffmpeg", "-y", "-i", main_ts, "-stream_loop", "-1", "-i", bgm,
         "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
         "-muxpreload", "0", "-muxdelay", "0", "-f", "mpegts", out_ts],
        label="(본편 배경음악)",
    )


def add_cover(video_mp4: str, thumb: str, out_mp4: str) -> None:
    """mp4 에 썸네일을 표지(커버 아트)로 삽입한다."""
    run_ffmpeg(
        ["ffmpeg", "-y", "-i", video_mp4, "-i", thumb,
         "-map", "0", "-map", "1", "-c", "copy",
         "-disposition:v:1", "attached_pic",
         "-movflags", "+faststart", out_mp4],
        label="(표지)",
    )


def finalize(video: str, srt: str, thumb: str, out_path: str,
             intro_sec: float = 2.5, add_intro: bool = True,
             cover: bool = True, burn: bool = True,
             font: str = "Paperlogy", font_size: int = 24,
             intro_video: str = "", outro_video: str = "",
             bgm: str = "", bgm_volume: float = 0.25,
             watermark: str = "", wm_pos: str = "tr",
             wm_scale: float = 0.12, wm_margin: int = 24,
             wm_colorkey: str = "", auto_labels: bool = False,
             gemini_key: str = "", gemini_model: str = GEMINI_MODEL,
             label_size: int = 44, loudnorm: bool = False,
             teaser_cuts: int = 0, teaser_sec: float = 1.5,
             impact_subs: str = "none", impact_size: int = 64,
             impact_color: str = "yellow", impact_pos: str = "center",
             impact_pop: bool = True) -> None:
    video = os.path.abspath(video)
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    w, h, fps = get_video_props(video)
    print(f"[정보] 영상 규격: {w}x{h}, {fps} fps", flush=True)

    # AI 자동 키워드: 자막(SRT)을 Gemini 에 보내 구간별 키워드를 받아 마크 아래 표시
    labels = []
    if auto_labels:
        if not gemini_key:
            print("[AI 키워드] 건너뜀: Gemini API 키가 없습니다.", flush=True)
        elif not (srt and os.path.isfile(srt)):
            print("[AI 키워드] 건너뜀: 자막(SRT) 파일이 필요합니다. "
                  "완성 탭에서 자막 파일을 선택하세요.", flush=True)
        else:
            print(f"[AI 키워드] Gemini 로 구간별 키워드 생성 중...", flush=True)
            labels = gemini_labels(srt, gemini_key, gemini_model)
            if not labels:
                print("[AI 키워드] 키워드를 만들지 못했습니다(응답 확인). "
                      "라벨 없이 계속합니다.", flush=True)

    with tempfile.TemporaryDirectory(prefix="finalize_") as tmp:
        ts_files = []

        # 0) 인트로 티저 (선택) — 맨 맨 앞 (cold-open)
        if teaser_cuts > 0:
            print(f"[인트로 티저] 본편에서 {teaser_cuts}개 피크 선정 중 (각 {teaser_sec}초)...", flush=True)
            teaser_files = _extract_teaser_clips(video, w, h, fps, tmp, n_clips=teaser_cuts,
                                               clip_duration=teaser_sec)
            if teaser_files:
                # 티저 끝에 화이트 플래시 + 오디오 fade 효과 추가 (0.25초)
                # 간단히: 마지막 커트에 fade-out 효과를 각각에 붙이고,
                # 마지막 커트 끝에 0.25초 화이트 프레임 추가
                last_idx = len(teaser_files) - 1
                for i, ts in enumerate(teaser_files):
                    if i == last_idx:
                        # 마지막 컷: fade + white flash
                        # st는 페이드 시작 시간 = (컷 길이 - 0.25초)
                        fade_start = teaser_sec - 0.25
                        faded_ts = os.path.join(tmp, f"teaser_fade_{i}.ts")
                        cmd = ["ffmpeg", "-y", "-i", ts,
                               "-vf", f"fade=t=out:st={fade_start}:d=0.25:color=white",
                               "-af", f"afade=t=out:st={fade_start}:d=0.25"] + _enc_opts(fps) + [faded_ts]
                        run_ffmpeg(cmd, label=f"(티저 페이드 {i+1})")
                        teaser_files[i] = faded_ts
                ts_files.extend(teaser_files)

        # 1) 인트로 영상 (있으면 티저 다음)
        if intro_video:
            print(f"[인트로 영상] 규격 맞추는 중...", flush=True)
            iv_ts = os.path.join(tmp, "intro_video.ts")
            prep_clip(intro_video, w, h, fps, iv_ts, label="(인트로 영상)")
            ts_files.append(iv_ts)

        # 2) 썸네일 인트로
        if add_intro and thumb:
            print(f"[썸네일 인트로] 생성 ({intro_sec}초)...", flush=True)
            intro_ts = os.path.join(tmp, "intro.ts")
            make_intro(thumb, w, h, fps, intro_sec, intro_ts)
            ts_files.append(intro_ts)

        # 3) 본편 (자막 하드섭 + 채널 마크 + AI 키워드는 여기서만 = 본영상에만)
        impact_cues = []
        impact_color_hex = ""

        # 임팩트 줄 선정 (impact_subs가 지정되고 burn and srt인 경우)
        if impact_subs != "none" and burn and srt and os.path.isfile(srt):
            print(f"[임팩트 자막] 오디오 에너지 분석 중...", flush=True)
            # 오디오 추출
            wav_path = os.path.join(tmp, "impact_audio.wav")
            extract_audio(video, wav_path)

            # 에너지 계산
            energy_arr, window_sec = compute_energy(wav_path)
            voice_arr, _ = compute_voice_energy(wav_path, tmp)

            # 임팩트 줄 선정
            cues = parse_srt(srt)
            impact_cues = select_impact_cues(cues, energy_arr, voice_arr, window_sec,
                                            impact_subs, min_gap=8.0, max_len=6.0)

            # 색상 코드 변환 (ASS &HAABBGGRR)
            color_map = {
                "yellow": "&H0000FFFF",
                "white": "&H00FFFFFF",
                "red": "&H000000FF",
                "cyan": "&H00FFFF00"
            }
            impact_color_hex = color_map.get(impact_color, "&H0000FFFF")

        wm_note = f" + 마크({WM_POSITIONS.get(wm_pos, wm_pos)})" if watermark else ""
        lb_note = f" + AI키워드({len(labels)})" if labels else ""
        impact_note = f" + 임팩트자막({len(impact_cues)})" if impact_cues else ""
        print(f"[본편] 처리{' + 자막 새겨넣기' if (burn and srt) else ''}{wm_note}{lb_note}{impact_note}...",
              flush=True)
        main_ts = os.path.join(tmp, "main.ts")
        srt_name = ""
        if burn and srt:
            srt_name = "sub.srt"
            shutil.copyfile(srt, os.path.join(tmp, srt_name))
        # Copy bundled fonts to tmpdir so libass can find them
        copy_fonts_to(tmp)
        render_main(video, srt_name, w, h, fps, main_ts, tmp, font, font_size,
                    burn_sub=bool(burn and srt), watermark=watermark,
                    wm_pos=wm_pos, wm_scale=wm_scale, wm_margin=wm_margin,
                    wm_colorkey=wm_colorkey, labels=labels, label_size=label_size,
                    impact_cues=impact_cues, impact_size=impact_size,
                    impact_color=impact_color_hex, impact_pos=impact_pos,
                    impact_pop=impact_pop)

        # 3-2) 배경음악: 본편에만 섞는다(인트로/아웃트로/썸네일엔 안 깔림).
        if bgm and has_audio(main_ts):
            print(f"[배경음악] 본편 구간에만 삽입 (볼륨 {bgm_volume})...", flush=True)
            main_bgm_ts = os.path.join(tmp, "main_bgm.ts")
            mix_bgm_into_ts(main_ts, bgm, main_bgm_ts, bgm_volume)
            ts_files.append(main_bgm_ts)
        else:
            if bgm:
                print(f"[배경음악] 본편에 오디오가 없어 건너뜁니다.", flush=True)
            ts_files.append(main_ts)

        # 4) 아웃트로 영상 (있으면 맨 뒤)
        if outro_video:
            print(f"[아웃트로 영상] 규격 맞추는 중...", flush=True)
            ov_ts = os.path.join(tmp, "outro_video.ts")
            prep_clip(outro_video, w, h, fps, ov_ts, label="(아웃트로 영상)")
            ts_files.append(ov_ts)

        # 5) 이어붙이기
        print(f"[이어붙이기] 조각 합치는 중...", flush=True)
        combined = os.path.join(tmp, "combined.mp4")
        concat_ts(ts_files, combined, tmp)
        stage = combined

        # 5-1) 음량 정규화 (유튜브 -14 LUFS, 2패스로 정확히)
        if loudnorm:
            print("[음량 정규화] 유튜브 표준(-14 LUFS)으로 2패스 정규화 중...", flush=True)
            loudnorm_out = os.path.join(tmp, "loudnorm.mp4")
            _loudnorm(stage, loudnorm_out)
            stage = loudnorm_out

        # 6) 표지(커버) 또는 최종 저장
        if cover and thumb:
            print(f"[표지] 썸네일 커버 삽입...", flush=True)
            add_cover(stage, thumb, out_path)
        else:
            shutil.copyfile(stage, out_path)

    print(f"\n완료! 저장됨: {out_path}", flush=True)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="영상+자막+썸네일 → 완성 영상")
    ap.add_argument("video", help="영상 파일 (mp4 등)")
    ap.add_argument("srt", help="자막 파일 (.srt)")
    ap.add_argument("thumb", help="썸네일 이미지 (jpg/png)")
    ap.add_argument("-o", "--output", required=True, help="출력 mp4 경로")
    ap.add_argument("--intro-sec", type=float, default=2.5, help="인트로 길이(초)")
    ap.add_argument("--intro-video", default="", help="맨 앞에 붙일 인트로 영상 파일")
    ap.add_argument("--outro-video", default="", help="맨 뒤에 붙일 아웃트로 영상 파일")
    ap.add_argument("--bgm", default="", help="배경음악 파일 (mp3/m4a/wav 등)")
    ap.add_argument("--bgm-volume", type=float, default=0.25,
                    help="배경음악 볼륨 배율(0~1, 기본 0.25)")
    ap.add_argument("--no-intro", dest="intro", action="store_false",
                    help="썸네일 인트로를 붙이지 않음")
    ap.add_argument("--no-cover", dest="cover", action="store_false",
                    help="썸네일 표지(커버)를 넣지 않음")
    ap.add_argument("--no-subs", dest="burn", action="store_false",
                    help="자막을 새겨넣지 않음")
    ap.add_argument("--font", default="Paperlogy", help="자막 글꼴")
    ap.add_argument("--font-size", type=int, default=24, help="자막 크기")
    ap.add_argument("--watermark", default="",
                    help="본영상에 새겨넣을 채널 마크(로고) 이미지 경로. "
                         "인트로/아웃트로엔 들어가지 않습니다.")
    ap.add_argument("--wm-pos", default="tr", choices=list(WM_POSITIONS.keys()),
                    help="마크 위치: tl(좌상) tr(우상) bl(좌하) br(우하). 기본 tr")
    ap.add_argument("--wm-scale", type=float, default=0.12,
                    help="마크 가로폭 = 영상 가로폭 * 이 값 (기본 0.12)")
    ap.add_argument("--wm-margin", type=int, default=24,
                    help="마크 가장자리 여백(픽셀). 기본 24")
    ap.add_argument("--wm-colorkey", default="",
                    help="마크 이미지의 배경색을 투명 처리(예: black / white / 0xRRGGBB). "
                         "비우면 이미지 그대로 사용. 단색 배경 로고에 유용.")
    ap.add_argument("--auto-labels", action="store_true",
                    help="자막(SRT)을 Gemini 에 보내 구간별 키워드를 받아 마크 아래 표시")
    ap.add_argument("--gemini-key", default="",
                    help="Google Gemini API 키 (--auto-labels 사용 시 필요)")
    ap.add_argument("--gemini-model", default=GEMINI_MODEL,
                    help=f"Gemini 모델명 (기본 {GEMINI_MODEL})")
    ap.add_argument("--label-size", type=int, default=44,
                    help="AI 키워드 라벨 글자 크기 (기본 44)")
    ap.add_argument("--loudnorm", action="store_true",
                    help="음량을 유튜브 표준(-14 LUFS)으로 정규화")
    ap.add_argument("--cpu-encode", action="store_true",
                    help="GPU 가속 인코딩 끄기 (호환성 문제 시)")
    ap.add_argument("--teaser", type=int, default=0,
                    help="인트로 티저: 본편에서 추출할 하이라이트 컷 수 (0=끔, 2~4 권장)")
    ap.add_argument("--teaser-sec", type=float, default=1.5,
                    help="티저 컷 하나의 길이(초) (기본 1.5, 범위 0.5~5.0)")
    ap.add_argument("--impact-subs", choices=["none", "low", "mid", "high"], default="none",
                    help="예능 자막(임팩트): none(끔), low(상위 5%), mid(상위 10%), high(상위 20%)")
    ap.add_argument("--impact-size", type=int, default=64,
                    help="임팩트 자막 크기 (픽셀, 기본 64, 범위 24~120)")
    ap.add_argument("--impact-color", choices=["yellow", "white", "red", "cyan"], default="yellow",
                    help="임팩트 자막 색상 (기본 yellow)")
    ap.add_argument("--impact-pos", choices=["center", "top", "bottom"], default="center",
                    help="임팩트 자막 위치 (기본 center)")
    ap.add_argument("--no-impact-pop", dest="impact_pop", action="store_false",
                    help="임팩트 자막 팝 효과 끄기 (기본 켜짐)")
    ap.set_defaults(intro=True, cover=True, burn=True, impact_pop=True)
    args = ap.parse_args()

    if args.cpu_encode:
        set_hw_encoding(False)

    # "-" 는 GUI 에서 '사용 안 함' 을 뜻하는 자리표시자.
    srt = "" if args.srt == "-" else args.srt
    thumb = "" if args.thumb == "-" else args.thumb
    intro_video = "" if args.intro_video in ("", "-") else args.intro_video
    outro_video = "" if args.outro_video in ("", "-") else args.outro_video
    bgm = "" if args.bgm in ("", "-") else args.bgm

    checks = [("영상", args.video, True)]
    if args.burn:
        checks.append(("자막", srt, True))
    if args.intro or args.cover:
        checks.append(("썸네일", thumb, True))
    if intro_video:
        checks.append(("인트로 영상", intro_video, True))
    if outro_video:
        checks.append(("아웃트로 영상", outro_video, True))
    if bgm:
        checks.append(("배경음악", bgm, True))
    for label, path, required in checks:
        if required and (not path or not os.path.isfile(path)):
            print(f"ERROR: {label} 파일을 찾을 수 없습니다: {path}")
            sys.exit(1)

    watermark = "" if args.watermark in ("", "-") else args.watermark
    if watermark and not os.path.isfile(watermark):
        print(f"ERROR: 마크 이미지를 찾을 수 없습니다: {watermark}")
        sys.exit(1)

    # impact_size를 24~120으로 클램프
    impact_size = max(24, min(120, args.impact_size))

    finalize(args.video, srt, thumb, args.output,
             intro_sec=args.intro_sec, add_intro=args.intro,
             cover=args.cover, burn=args.burn,
             font=args.font, font_size=args.font_size,
             intro_video=intro_video, outro_video=outro_video,
             bgm=bgm, bgm_volume=args.bgm_volume,
             watermark=watermark, wm_pos=args.wm_pos,
             wm_scale=args.wm_scale, wm_margin=args.wm_margin,
             wm_colorkey=args.wm_colorkey,
             auto_labels=args.auto_labels, gemini_key=args.gemini_key,
             gemini_model=args.gemini_model, label_size=args.label_size,
             loudnorm=args.loudnorm, teaser_cuts=args.teaser,
             teaser_sec=args.teaser_sec,
             impact_subs=args.impact_subs, impact_size=impact_size,
             impact_color=args.impact_color, impact_pos=args.impact_pos,
             impact_pop=args.impact_pop)


if __name__ == "__main__":
    main()

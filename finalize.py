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


def _setup_bundled_paths():
    """포터블 배포용: 스크립트 폴더 옆의 ffmpeg/bin 을 PATH 에 추가."""
    base = os.path.dirname(os.path.abspath(__file__))
    for rel in (os.path.join("ffmpeg", "bin"), "ffmpeg"):
        p = os.path.join(base, rel)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "ffmpeg.exe")):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            break


_setup_bundled_paths()

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
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "23",
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


WM_POSITIONS = {"tl": "좌상단", "tr": "우상단", "bl": "좌하단", "br": "우하단"}

# ── AI 자동 키워드(Gemini) ─────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"


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
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:300]
        except Exception:
            pass
        print(f"  (Gemini 요청 실패 HTTP {e.code}: {detail})", flush=True)
        return []
    except Exception as e:
        print(f"  (Gemini 연결 실패: {e})", flush=True)
        return []
    try:
        data = json.loads(raw)
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        print(f"  (Gemini 응답 형식이 예상과 다름: {raw[:300]})", flush=True)
        return []
    labels = _parse_label_lines(text)
    if labels:
        print(f"  Gemini 키워드 {len(labels)}개 생성", flush=True)
    else:
        one = " / ".join(text.splitlines())[:200]
        print(f"  (Gemini 응답에서 키워드 파싱 실패. 실제 응답: {one})", flush=True)
    return labels


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


def render_main(video: str, srt_name: str, w: int, h: int, fps: str,
                out_ts: str, cwd: str, font: str, font_size: int,
                burn_sub: bool, watermark: str = "", wm_pos: str = "tr",
                wm_scale: float = 0.12, wm_margin: int = 24,
                wm_colorkey: str = "", labels=None,
                label_font: str = "Malgun Gothic", label_size: int = 44) -> None:
    """본편을 규격 통일 + (선택)자막 하드섭 + (선택)채널 마크 오버레이 하여 TS 로.

    마크는 본편에만 들어간다(인트로/아웃트로 TS 는 손대지 않으므로 자동으로
    본영상에만 남는다). subtitles 필터 경로는 Windows 이스케이프가 까다로워
    SRT 를 작업 폴더(cwd)에 복사한 뒤 상대 경로로 참조한다.

    wm_colorkey 를 주면(예: black/white/0xRRGGBB) 마크 이미지에서 그 배경색을
    투명 처리(colorkey)한 뒤 얹는다. 배경이 단색인 로고를 투명 없이 써도 된다.
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
        style = (f"FontName={font},FontSize={font_size},"
                 f"PrimaryColour=&H00FFFFFF,OutlineColour=&H90000000,"
                 f"BorderStyle=1,Outline=2,Shadow=1,MarginV=28")
        base += f",subtitles={srt_name}:force_style='{style}'"
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
        fc_parts.append(f"{cur}subtitles=labels.ass[vlab]")
        cur = "[vlab]"

    # 4) 마무리 포맷
    fc_parts.append(f"{cur}format=yuv420p[v]")
    fc = ";".join(fc_parts)

    cmd = (["ffmpeg", "-y"] + inputs +
           ["-filter_complex", fc, "-map", "[v]", "-map", "0:a?"] +
           _enc_opts(fps) + [out_ts])
    run_ffmpeg(cmd, label="(본편)", cwd=cwd)


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
             font: str = "Malgun Gothic", font_size: int = 24,
             intro_video: str = "", outro_video: str = "",
             bgm: str = "", bgm_volume: float = 0.25,
             watermark: str = "", wm_pos: str = "tr",
             wm_scale: float = 0.12, wm_margin: int = 24,
             wm_colorkey: str = "", auto_labels: bool = False,
             gemini_key: str = "", gemini_model: str = GEMINI_MODEL,
             label_size: int = 44) -> None:
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

        # 1) 인트로 영상 (있으면 맨 앞)
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
        wm_note = f" + 마크({WM_POSITIONS.get(wm_pos, wm_pos)})" if watermark else ""
        lb_note = f" + AI키워드({len(labels)})" if labels else ""
        print(f"[본편] 처리{' + 자막 새겨넣기' if (burn and srt) else ''}{wm_note}{lb_note}...",
              flush=True)
        main_ts = os.path.join(tmp, "main.ts")
        srt_name = ""
        if burn and srt:
            srt_name = "sub.srt"
            shutil.copyfile(srt, os.path.join(tmp, srt_name))
        render_main(video, srt_name, w, h, fps, main_ts, tmp, font, font_size,
                    burn_sub=bool(burn and srt), watermark=watermark,
                    wm_pos=wm_pos, wm_scale=wm_scale, wm_margin=wm_margin,
                    wm_colorkey=wm_colorkey, labels=labels, label_size=label_size)

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

        # 6) 표지(커버) 또는 최종 저장
        if cover and thumb:
            print(f"[표지] 썸네일 커버 삽입...", flush=True)
            add_cover(stage, thumb, out_path)
        else:
            shutil.copyfile(stage, out_path)

    print(f"\n완료! 저장됨: {out_path}", flush=True)


def main():
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
    ap.add_argument("--font", default="Malgun Gothic", help="자막 글꼴")
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
    ap.set_defaults(intro=True, cover=True, burn=True)
    args = ap.parse_args()

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
             gemini_model=args.gemini_model, label_size=args.label_size)


if __name__ == "__main__":
    main()

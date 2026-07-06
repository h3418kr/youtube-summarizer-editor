"""GUI launcher for summarizer.py (라이브 하이라이트 메이커 / Live Highlight Maker)

네 개의 탭 / Four tabs:
  1. 영상 요약 / Summarize   — URL·로컬 파일 → 요약 영상 + 자막(SRT)
  2. 수동 하이라이트 / Manual — 로컬 영상 + 직접 입력한 시간대 → 하이라이트 영상
  3. 쇼츠 만들기 / Shorts     — 구간을 골라 9:16 세로 쇼츠 영상 (shorts.py)
  4. 완성 영상 만들기 / Finalize — 영상 + 자막 + 썸네일 → 완성 mp4 (finalize.py)

한/영 전환 버튼 제공 / KO-EN language toggle button.
"""
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 내부 코드값 (언어와 무관) / internal codes (language-independent) ──────────────
MODEL_CODES = ["tiny", "base", "small", "medium", "large"]
QUALITY_CODES = ["360", "480", "720", "1080"]
TRANS_CODES = ["none", "black", "white"]
SFX_CODES = ["none", "whoosh", "swoosh", "beep", "pop", "impact"]
WMPOS_CODES = ["tl", "tr", "bl", "br"]  # 좌상 우상 좌하 우하
WMKEY_CODES = ["", "white", "black"]  # 배경 투명 처리: 없음 / 흰색 / 검정

# Gemini API 키를 저장/불러오기 (한 번 입력하면 다음에도 자동 채움)
GEMINI_KEY_FILE = os.path.join(SCRIPT_DIR, "gemini_key.txt")


def load_gemini_key():
    try:
        with open(GEMINI_KEY_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def save_gemini_key(key):
    try:
        with open(GEMINI_KEY_FILE, "w", encoding="utf-8") as f:
            f.write((key or "").strip())
    except Exception:
        pass
SHORTS_MODE_CODES = ["center", "blur"]
SHORTS_SUBPOS_CODES = ["bottom", "center", "top"]  # 하단 중앙 상단

# 분석 전용 모드 결과를 수동 하이라이트 탭으로 넘길 때 쓰는 위젯 참조
MANUAL_TAB = {}

# ── 번역 문자열 / translation strings ─────────────────────────────────────────────
STRINGS = {
    "ko": {
        "win_title": "라이브 하이라이트 메이커",
        "tab_summarize": "  영상 요약  ",
        "tab_finalize": "  완성 영상 만들기  ",
        "lang_button": "🌐 English",
        # summarize tab
        "sum_heading": "영상 요약 — 방송을 하이라이트로",
        "url": "URL 또는 로컬 파일",
        "browse_file": "파일 선택",
        "dlg_url_video": "로컬 영상 파일 선택",
        "url_hint": "(방송 URL을 넣거나, 직접 녹화한 로컬 영상 파일을 선택하세요)",
        "outdir": "출력 폴더",
        "browse": "찾아보기",
        "options": "옵션",
        "target_min": "목표 길이 (분)",
        "model": "Whisper 모델 (자막 품질)",
        "model_values": ["tiny (빠름)", "base", "small (권장)", "medium (정확)", "large (최고)"],
        "language": "언어",
        "quality": "화질",
        "quality_values": QUALITY_CODES,
        "expand_before": "피크 앞 확장 (초)",
        "expand_after": "피크 뒤 확장 (초)",
        "bridge": "같은 장면 묶기 기준 (초)",
        "bridge_hint": "(시간차가 이보다 짧으면 한 장면으로 이어붙임)",
        "transition": "화면 전환",
        "transition_values": ["없음", "암전 (기본)", "화이트 플래시"],
        "sfx": "전환 효과음",
        "sfx_values": ["없음", "휙 (기본)", "스와이프", "삑", "팝", "임팩트"],
        "sfx_hint": "(서로 다른 하이라이트 사이에 적용됩니다)",
        "btn_summarize": "다운로드 & 요약",
        "processing": "처리 중...",
        "save_video": "원본 영상 보관 폴더 (선택)",
        "save_video_hint": "(지정하면 다운로드한 원본을 삭제하지 않고 이 폴더에 보관합니다)",
        "dlg_save_video": "원본 영상 보관 폴더 선택",
        "watermark": "채널 마크 이미지 (선택)",
        "watermark_hint": "(본영상에만 표시 — 완성 탭의 인트로/아웃트로엔 안 나옵니다)",
        "dlg_watermark": "마크 이미지 선택 (png/jpg)",
        "wm_position": "마크 위치",
        "wm_pos_values": ["좌상단", "우상단", "좌하단", "우하단"],
        "wm_colorkey": "배경 투명 처리",
        "wm_colorkey_values": ["없음", "흰색 제거", "검정 제거"],
        "wm_colorkey_hint": "(단색 배경 로고면 그 색을 투명 처리)",
        "auto_labels": "AI 자동 키워드 (Gemini)",
        "auto_labels_hint": "(자막을 AI로 분석해 구간별 키워드를 마크 아래 표시)",
        "gemini_key": "Gemini API 키",
        "gemini_key_hint": "(aistudio.google.com 에서 무료 발급 · 저장됨)",
        "label_position": "소제목 위치",
        "label_position_hint": "(‘시작-끝 | 소제목’으로 입력한 소제목이 뜨는 자리)",
        "analyze_only": "구간 후보만 분석 (영상은 만들지 않음 — 빠름)",
        "analyze_hint": "(분석이 끝나면 후보 구간이 수동 하이라이트 탭에 자동으로 채워집니다)",
        "msg_analyze_done": ("하이라이트 후보 분석 완료!\n\n"
                             "후보 구간 목록을 수동 하이라이트 탭에 불러왔습니다.\n"
                             "구간을 확인·수정한 뒤 「하이라이트 만들기」를 누르세요."),
        # manual highlight tab
        "tab_manual": "  수동 하이라이트  ",
        "man_heading": "수동 하이라이트 — 받아둔 영상으로 직접 편집",
        "man_video": "영상 파일 (로컬)",
        "man_ranges": "하이라이트 시간대",
        "man_ranges_hint": "한 줄에 하나씩:  시작 - 끝  |  소제목(선택)   (예: 1:23 - 2:05 | 다운그레이드)",
        "man_ranges_example": ("# 예시입니다. 이 줄을 지우고 시간대를 입력하세요.\n"
                               "# '|' 뒤에 소제목을 적으면 마크 아래에 표시됩니다(선택).\n"
                               "1:23 - 2:05 | 다운그레이드\n5:40 - 6:10\n"),
        "man_name": "출력 이름 (선택)",
        "man_subtitles": "자막(SRT) 자동 생성 (Whisper, 느림)",
        "btn_manual": "하이라이트 만들기",
        "dlg_man_video": "영상 파일 선택",
        "msg_need_man_video": "편집할 영상 파일을 선택하세요.",
        "msg_need_ranges": "하이라이트 시간대를 한 줄 이상 입력하세요.\n예: 1:23 - 2:05",
        "msg_man_done": ("하이라이트 영상이 저장되었습니다.\n\n폴더: {folder}\n\n"
                         "• _highlight.mp4  — 하이라이트 영상\n"
                         "• _chapters.txt   — 유튜브 챕터 (설명란에 붙여넣기)"),
        # shorts tab
        "tab_shorts": "  쇼츠 만들기  ",
        "shorts_heading": "쇼츠 만들기 — 구간을 골라 9:16 세로 영상으로",
        "shorts_ranges": "쇼츠 구간",
        "shorts_ranges_hint": "한 줄에 하나씩:  시작 - 끝   (총 3분 이하 권장, 여러 줄은 하드컷으로 이어붙임)",
        "shorts_ranges_example": "# 쇼츠로 만들 구간을 입력하세요. (예: 하이라이트 한 장면)\n0:10 - 0:40\n",
        "shorts_mode": "세로 변환 방식",
        "shorts_mode_values": ["중앙 크롭 (기본)", "블러 배경"],
        "shorts_mode_hint": "(중앙 크롭: 화면 가운데 확대 / 블러 배경: 원본 그대로 + 위아래 블러)",
        "shorts_subtitles": "자막 자동 생성해 크게 새겨넣기 (Whisper)",
        "shorts_font_size": "자막 크기",
        "shorts_sub_pos": "자막 위치",
        "shorts_sub_pos_values": ["하단 (기본)", "중앙", "상단"],
        "btn_shorts": "쇼츠 만들기",
        "msg_shorts_done": ("쇼츠 영상이 저장되었습니다.\n\n폴더: {folder}\n\n"
                            "• _shorts.mp4  — 1080x1920 세로 영상"),
        # finalize tab
        "video": "영상 파일",
        "srt": "자막 파일 (.srt)",
        "thumb": "썸네일 이미지",
        "intro_video": "인트로 영상 (선택)",
        "outro_video": "아웃트로 영상 (선택)",
        "bgm": "배경음악 (선택)",
        "outfile": "출력 파일",
        "save_as": "저장 위치",
        "opt_intro": "썸네일 인트로 붙이기",
        "intro_sec": "인트로 길이 (초)",
        "opt_cover": "썸네일 표지(커버) 삽입",
        "font_size": "자막 크기",
        "opt_burn": "자막 영상에 새겨넣기(하드섭)",
        "bgm_volume": "배경음악 볼륨 (0~1)",
        "btn_finalize": "완성 영상 만들기",
        # file dialog titles
        "dlg_outdir": "출력 폴더 선택",
        "dlg_video": "영상 파일 선택",
        "dlg_srt": "자막 파일 선택",
        "dlg_thumb": "썸네일 이미지 선택",
        "dlg_intro": "인트로 영상 선택",
        "dlg_outro": "아웃트로 영상 선택",
        "dlg_bgm": "배경음악 파일 선택",
        "dlg_save": "완성 영상 저장",
        # filetype labels
        "ft_video": "동영상",
        "ft_audio": "오디오",
        "ft_subtitle": "자막",
        "ft_image": "이미지",
        "ft_mp4": "MP4 영상",
        "ft_all": "전체",
        # messages
        "msg_input_error": "입력 오류",
        "msg_need_url": "영상/방송 URL을 입력하세요.",
        "msg_done": "완료",
        "msg_summ_done": ("저장 완료!\n\n폴더: {folder}\n\n"
                          "• _summary.mp4   — 요약 영상\n"
                          "• _summary.srt   — 자막 파일 (편집 후 영상에 적용)\n"
                          "• _chapters.txt  — 유튜브 챕터 (설명란에 붙여넣기)"),
        "msg_error": "오류",
        "msg_error_body": "처리 중 오류가 발생했습니다.\n로그를 확인하세요.",
        "msg_need_video": "영상 파일을 선택하세요.",
        "msg_need_srt": "자막 파일을 선택하세요.\n(자막 새겨넣기를 끄면 자막 없이 진행됩니다.)",
        "msg_need_srt_labels": "AI 자동 키워드를 쓰려면 자막(SRT) 파일이 필요합니다.\nAI가 자막 내용을 분석해 키워드를 만들기 때문입니다.",
        "msg_need_thumb": "썸네일 이미지를 선택하세요.\n(인트로/표지 옵션을 모두 끄면 썸네일 없이 진행됩니다.)",
        "msg_need_out": "출력 파일 경로를 지정하세요.",
        "msg_final_done": "완성 영상이 저장되었습니다.\n\n{path}",
    },
    "en": {
        "win_title": "Live Highlight Maker",
        "tab_summarize": "  Summarize  ",
        "tab_finalize": "  Finalize  ",
        "lang_button": "🌐 한국어",
        # summarize tab
        "sum_heading": "Summarize — turn broadcasts into highlights",
        "url": "URL or local file",
        "browse_file": "Pick file",
        "dlg_url_video": "Select local video file",
        "url_hint": "(paste a stream URL, or pick a video file you recorded yourself)",
        "outdir": "Output folder",
        "browse": "Browse",
        "options": "Options",
        "target_min": "Target length (min)",
        "model": "Whisper model (subtitle quality)",
        "model_values": ["tiny (fast)", "base", "small (recommended)", "medium (accurate)", "large (best)"],
        "language": "Language",
        "quality": "Quality",
        "quality_values": QUALITY_CODES,
        "expand_before": "Expand before peak (s)",
        "expand_after": "Expand after peak (s)",
        "bridge": "Merge scenes within (s)",
        "bridge_hint": "(clips closer than this are joined into one)",
        "transition": "Transition",
        "transition_values": ["None", "Black (default)", "White flash"],
        "sfx": "Transition SFX",
        "sfx_values": ["None", "Whoosh (default)", "Swoosh", "Beep", "Pop", "Impact"],
        "sfx_hint": "(applied between different highlights)",
        "btn_summarize": "Download & Summarize",
        "processing": "Processing...",
        "save_video": "Keep original video in folder (optional)",
        "save_video_hint": "(if set, the downloaded original is kept here instead of deleted)",
        "dlg_save_video": "Select folder to keep original video",
        "watermark": "Channel mark image (optional)",
        "watermark_hint": "(shown on main video only — not on the Finalize intro/outro)",
        "dlg_watermark": "Select mark image (png/jpg)",
        "wm_position": "Mark position",
        "wm_pos_values": ["Top-left", "Top-right", "Bottom-left", "Bottom-right"],
        "wm_colorkey": "Make background transparent",
        "wm_colorkey_values": ["None", "Remove white", "Remove black"],
        "wm_colorkey_hint": "(for a logo on a solid-color background)",
        "auto_labels": "Auto keywords (Gemini AI)",
        "auto_labels_hint": "(AI reads the subtitles and shows a keyword per section under the mark)",
        "gemini_key": "Gemini API key",
        "gemini_key_hint": "(free key from aistudio.google.com · saved)",
        "label_position": "Subtitle position",
        "label_position_hint": "(where the 'start-end | subtitle' text appears)",
        "analyze_only": "Analyze candidates only (no video output — fast)",
        "analyze_hint": "(when done, the candidate ranges are loaded into the Manual highlights tab)",
        "msg_analyze_done": ("Highlight analysis finished!\n\n"
                             "The candidate ranges were loaded into the Manual highlights tab.\n"
                             "Review / adjust them, then click \"Make highlights\"."),
        # manual highlight tab
        "tab_manual": "  Manual highlights  ",
        "man_heading": "Manual highlights — edit a video you already have",
        "man_video": "Video file (local)",
        "man_ranges": "Highlight time ranges",
        "man_ranges_hint": "One per line:  start - end  |  subtitle (optional)   (e.g. 1:23 - 2:05 | Downgrade)",
        "man_ranges_example": ("# Example. Delete this line and enter your own ranges.\n"
                               "# Text after '|' shows as a subtitle under the mark (optional).\n"
                               "1:23 - 2:05 | Downgrade\n5:40 - 6:10\n"),
        "man_name": "Output name (optional)",
        "man_subtitles": "Auto-generate subtitles (SRT) with Whisper (slow)",
        "btn_manual": "Make highlights",
        "dlg_man_video": "Select video file",
        "msg_need_man_video": "Please select a video file to edit.",
        "msg_need_ranges": "Enter at least one highlight time range.\nExample: 1:23 - 2:05",
        "msg_man_done": ("Highlight video saved.\n\nFolder: {folder}\n\n"
                         "- _highlight.mp4  — highlight video\n"
                         "- _chapters.txt   — YouTube chapters (paste into description)"),
        # shorts tab
        "tab_shorts": "  Shorts  ",
        "shorts_heading": "Shorts — pick ranges, get a 9:16 vertical video",
        "shorts_ranges": "Shorts ranges",
        "shorts_ranges_hint": "One per line:  start - end   (3 min total max recommended; lines are hard-cut together)",
        "shorts_ranges_example": "# Enter the range(s) for the short (e.g. one highlight scene).\n0:10 - 0:40\n",
        "shorts_mode": "Vertical mode",
        "shorts_mode_values": ["Center crop (default)", "Blur background"],
        "shorts_mode_hint": "(center crop: zoom into the middle / blur: full frame + blurred bars)",
        "shorts_subtitles": "Auto-generate big burned-in subtitles (Whisper)",
        "shorts_font_size": "Subtitle size",
        "shorts_sub_pos": "Subtitle position",
        "shorts_sub_pos_values": ["Bottom (default)", "Center", "Top"],
        "btn_shorts": "Make Short",
        "msg_shorts_done": ("Short saved.\n\nFolder: {folder}\n\n"
                            "- _shorts.mp4  — 1080x1920 vertical video"),
        # finalize tab
        "video": "Video file",
        "srt": "Subtitle (.srt)",
        "thumb": "Thumbnail image",
        "intro_video": "Intro video (optional)",
        "outro_video": "Outro video (optional)",
        "bgm": "Background music (optional)",
        "outfile": "Output file",
        "save_as": "Save as",
        "opt_intro": "Add thumbnail intro",
        "intro_sec": "Intro length (s)",
        "opt_cover": "Embed thumbnail cover",
        "font_size": "Subtitle size",
        "opt_burn": "Burn subtitles (hardsub)",
        "bgm_volume": "BGM volume (0-1)",
        "btn_finalize": "Make Final Video",
        # file dialog titles
        "dlg_outdir": "Select output folder",
        "dlg_video": "Select video file",
        "dlg_srt": "Select subtitle file",
        "dlg_thumb": "Select thumbnail image",
        "dlg_intro": "Select intro video",
        "dlg_outro": "Select outro video",
        "dlg_bgm": "Select background music",
        "dlg_save": "Save final video",
        # filetype labels
        "ft_video": "Video",
        "ft_audio": "Audio",
        "ft_subtitle": "Subtitle",
        "ft_image": "Image",
        "ft_mp4": "MP4 video",
        "ft_all": "All files",
        # messages
        "msg_input_error": "Input error",
        "msg_need_url": "Please enter a video / stream URL.",
        "msg_done": "Done",
        "msg_summ_done": ("Saved!\n\nFolder: {folder}\n\n"
                          "- _summary.mp4   — summary video\n"
                          "- _summary.srt   — subtitle file (edit, then apply to video)\n"
                          "- _chapters.txt  — YouTube chapters (paste into description)"),
        "msg_error": "Error",
        "msg_error_body": "An error occurred during processing.\nCheck the log.",
        "msg_need_video": "Please select a video file.",
        "msg_need_srt": "Please select a subtitle file.\n(Turn off subtitle burning to proceed without subtitles.)",
        "msg_need_srt_labels": "Auto keywords need a subtitle (SRT) file.\nThe AI reads the subtitle content to make the keywords.",
        "msg_need_thumb": "Please select a thumbnail image.\n(Turn off both intro/cover options to proceed without a thumbnail.)",
        "msg_need_out": "Please specify an output file path.",
        "msg_final_done": "Final video saved.\n\n{path}",
    },
}

STATE = {"lang": "ko"}
_i18n = []  # registered widgets: (kind, obj, key)


def _t(key):
    return STRINGS[STATE["lang"]].get(key, key)


def reg(kind, obj, key):
    _i18n.append((kind, obj, key))
    return obj


def apply_language():
    for kind, obj, key in _i18n:
        if kind == "text":
            obj.config(text=_t(key))
        elif kind == "var":
            obj.set(_t(key))
        elif kind == "tab":
            nb, tab_id = obj
            nb.tab(tab_id, text=_t(key))
        elif kind == "combo":
            combo, values_key = obj
            idx = combo.current()
            combo["values"] = _t(values_key)
            combo.current(idx if idx >= 0 else 0)
        elif kind == "title":
            obj.title(_t(key))


def run_script(cmd, log_widget, done_cb):
    q = queue.Queue()

    def worker():
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=SCRIPT_DIR,
            )
            for line in proc.stdout:
                q.put(("log", line))
            proc.wait()
            q.put(("done", proc.returncode == 0))
        except Exception as e:
            q.put(("log", f"[오류] {e}\n"))
            q.put(("done", False))

    def poll():
        try:
            while True:
                msg_type, payload = q.get_nowait()
                if msg_type == "log":
                    log_widget.config(state="normal")
                    log_widget.insert(tk.END, payload)
                    log_widget.see(tk.END)
                    log_widget.config(state="disabled")
                elif msg_type == "done":
                    done_cb(payload)
                    return
        except queue.Empty:
            pass
        log_widget.after(100, poll)

    threading.Thread(target=worker, daemon=True).start()
    log_widget.after(100, poll)


def make_log(parent):
    return scrolledtext.ScrolledText(
        parent, height=14, state="disabled",
        bg="#1e1e1e", fg="#d4d4d4",
        font=("Consolas", 9), relief="flat",
    )


def _label(parent, key, **grid):
    w = ttk.Label(parent, text=_t(key))
    reg("text", w, key)
    if grid:
        w.grid(**grid)
    return w


def _load_analysis_into_manual(log_widget):
    """분석 전용 모드 로그에서 결과 파일 경로를 찾아 수동 하이라이트 탭에 채운다.

    summarizer.py 가 출력하는 'SEGMENTS_FILE::경로' / 'SOURCE_VIDEO::경로'
    마커 줄을 파싱한다. 성공하면 True."""
    if not MANUAL_TAB:
        return False
    txt = log_widget.get("1.0", tk.END)
    seg_m = re.findall(r"SEGMENTS_FILE::(.+)", txt)
    src_m = re.findall(r"SOURCE_VIDEO::(.+)", txt)
    if not seg_m:
        return False
    try:
        with open(seg_m[-1].strip(), "r", encoding="utf-8") as f:
            ranges = f.read().strip()
    except OSError:
        return False
    if not ranges:
        return False
    MANUAL_TAB["ranges_text"].delete("1.0", tk.END)
    MANUAL_TAB["ranges_text"].insert("1.0", ranges + "\n")
    if src_m and src_m[-1].strip():
        MANUAL_TAB["video_var"].set(src_m[-1].strip())
    return True


# ── 탭 1: 영상 요약 ────────────────────────────────────────────────────────────

def build_summarizer_tab(nb):
    frame = ttk.Frame(nb, padding=4)
    nb.add(frame, text=_t("tab_summarize"))
    reg("tab", (nb, frame), "tab_summarize")
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(6, weight=1)

    pad = {"padx": 12, "pady": 4}

    heading = ttk.Label(frame, text=_t("sum_heading"), font=("Segoe UI", 13, "bold"))
    reg("text", heading, "sum_heading")
    heading.grid(row=0, column=0, columnspan=3, sticky="w", padx=12, pady=(10, 8))

    # URL 또는 로컬 파일
    _label(frame, "url", row=1, column=0, sticky="w", **pad)
    url_var = tk.StringVar()
    ttk.Entry(frame, textvariable=url_var, width=52).grid(
        row=1, column=1, sticky="ew", padx=(0, 4), pady=4)

    def pick_local_video():
        path = filedialog.askopenfilename(
            title=_t("dlg_url_video"),
            filetypes=[("Video", "*.mp4 *.mkv *.mov *.avi *.webm *.flv *.ts *.m4v"),
                       ("All files", "*.*")])
        if path:
            url_var.set(path)

    browse_file = ttk.Button(frame, text=_t("browse_file"), command=pick_local_video)
    reg("text", browse_file, "browse_file")
    browse_file.grid(row=1, column=2, padx=(0, 12), pady=4)

    _label(frame, "url_hint", row=2, column=1, columnspan=2, sticky="w", padx=(0, 12), pady=(0, 4))

    # 출력 폴더
    _label(frame, "outdir", row=3, column=0, sticky="w", **pad)
    outdir_var = tk.StringVar(value=os.path.join(SCRIPT_DIR, "output"))
    ttk.Entry(frame, textvariable=outdir_var, width=52).grid(
        row=3, column=1, sticky="ew", padx=(0, 4), pady=4)
    browse_out = ttk.Button(
        frame, text=_t("browse"),
        command=lambda: outdir_var.set(
            filedialog.askdirectory(title=_t("dlg_outdir")) or outdir_var.get()))
    reg("text", browse_out, "browse")
    browse_out.grid(row=3, column=2, padx=(0, 12), pady=4)

    # 옵션
    opt = ttk.LabelFrame(frame, text=_t("options"), padding=8)
    reg("text", opt, "options")
    opt.grid(row=4, column=0, columnspan=3, sticky="ew", padx=12, pady=6)
    opt.columnconfigure(1, weight=1)
    opt.columnconfigure(3, weight=1)

    target_var = tk.StringVar(value="10")
    lang_var = tk.StringVar(value="ko")
    before_var = tk.StringVar(value="5")
    after_var = tk.StringVar(value="20")
    bridge_var = tk.StringVar(value="8")

    _label(opt, "target_min", row=0, column=0, sticky="w", padx=(8, 4), pady=3)
    ttk.Entry(opt, textvariable=target_var, width=6).grid(row=0, column=1, sticky="w", pady=3)
    _label(opt, "model", row=0, column=2, sticky="w", padx=(16, 4), pady=3)
    model_combo = ttk.Combobox(opt, values=_t("model_values"), width=16, state="readonly")
    model_combo.current(2)  # small
    reg("combo", (model_combo, "model_values"), "model_values")
    model_combo.grid(row=0, column=3, sticky="w", pady=3)

    _label(opt, "language", row=1, column=0, sticky="w", padx=(8, 4), pady=3)
    ttk.Entry(opt, textvariable=lang_var, width=6).grid(row=1, column=1, sticky="w", pady=3)
    _label(opt, "quality", row=1, column=2, sticky="w", padx=(16, 4), pady=3)
    quality_combo = ttk.Combobox(opt, values=_t("quality_values"), width=6, state="readonly")
    quality_combo.current(2)  # 720
    reg("combo", (quality_combo, "quality_values"), "quality_values")
    quality_combo.grid(row=1, column=3, sticky="w", pady=3)

    _label(opt, "expand_before", row=2, column=0, sticky="w", padx=(8, 4), pady=3)
    ttk.Entry(opt, textvariable=before_var, width=6).grid(row=2, column=1, sticky="w", pady=3)
    _label(opt, "expand_after", row=2, column=2, sticky="w", padx=(16, 4), pady=3)
    ttk.Entry(opt, textvariable=after_var, width=6).grid(row=2, column=3, sticky="w", pady=3)

    _label(opt, "bridge", row=3, column=0, sticky="w", padx=(8, 4), pady=3)
    ttk.Entry(opt, textvariable=bridge_var, width=6).grid(row=3, column=1, sticky="w", pady=3)
    _label(opt, "bridge_hint", row=3, column=2, columnspan=2, sticky="w", padx=(16, 4), pady=3)

    _label(opt, "transition", row=4, column=0, sticky="w", padx=(8, 4), pady=(6, 3))
    trans_combo = ttk.Combobox(opt, values=_t("transition_values"), width=14, state="readonly")
    trans_combo.current(1)  # black
    reg("combo", (trans_combo, "transition_values"), "transition_values")
    trans_combo.grid(row=4, column=1, sticky="w", pady=(6, 3))
    _label(opt, "sfx", row=4, column=2, sticky="w", padx=(16, 4), pady=(6, 3))
    sfx_combo = ttk.Combobox(opt, values=_t("sfx_values"), width=14, state="readonly")
    sfx_combo.current(1)  # whoosh
    reg("combo", (sfx_combo, "sfx_values"), "sfx_values")
    sfx_combo.grid(row=4, column=3, sticky="w", pady=(6, 3))
    _label(opt, "sfx_hint", row=5, column=0, columnspan=4, sticky="w", padx=(8, 4), pady=(0, 3))

    # 원본 영상 보관 폴더 (선택)
    save_video_var = tk.StringVar(value="")
    _label(opt, "save_video", row=6, column=0, sticky="w", padx=(8, 4), pady=(6, 3))
    ttk.Entry(opt, textvariable=save_video_var, width=28).grid(
        row=6, column=1, columnspan=2, sticky="ew", pady=(6, 3))
    browse_save = ttk.Button(
        opt, text=_t("browse"),
        command=lambda: save_video_var.set(
            filedialog.askdirectory(title=_t("dlg_save_video")) or save_video_var.get()))
    reg("text", browse_save, "browse")
    browse_save.grid(row=6, column=3, sticky="w", padx=(8, 4), pady=(6, 3))
    _label(opt, "save_video_hint", row=7, column=0, columnspan=4, sticky="w", padx=(8, 4), pady=(0, 3))

    # 분석 전용 모드: 후보 구간만 뽑아 수동 하이라이트 탭으로 넘긴다
    analyze_var = tk.BooleanVar(value=False)
    chk_analyze = ttk.Checkbutton(opt, text=_t("analyze_only"), variable=analyze_var)
    reg("text", chk_analyze, "analyze_only")
    chk_analyze.grid(row=8, column=0, columnspan=4, sticky="w", padx=8, pady=(6, 0))
    _label(opt, "analyze_hint", row=9, column=0, columnspan=4, sticky="w", padx=(8, 4), pady=(0, 3))

    # 실행 버튼
    btn_label_var = tk.StringVar(value=_t("btn_summarize"))
    reg("var", btn_label_var, "btn_summarize")
    run_btn = ttk.Button(frame, textvariable=btn_label_var, style="Accent.TButton")
    run_btn.grid(row=5, column=0, columnspan=3, padx=12, pady=8, sticky="new")

    # 로그
    log = make_log(frame)
    log.grid(row=6, column=0, columnspan=3, sticky="nsew", padx=12, pady=(0, 12))

    def on_run():
        url = url_var.get().strip()
        if not url:
            messagebox.showwarning(_t("msg_input_error"), _t("msg_need_url"))
            return

        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "summarizer.py"),
            url,
            "--target-min", target_var.get(),
            "--model", MODEL_CODES[max(model_combo.current(), 0)],
            "--lang", lang_var.get(),
            "--expand-before", before_var.get(),
            "--expand-after", after_var.get(),
            "--output-dir", outdir_var.get(),
            "--max-height", QUALITY_CODES[max(quality_combo.current(), 0)],
            "--bridge-gap", bridge_var.get(),
            "--transition-style", TRANS_CODES[max(trans_combo.current(), 0)],
            "--sfx", SFX_CODES[max(sfx_combo.current(), 0)],
        ]
        if save_video_var.get().strip():
            cmd += ["--save-video", save_video_var.get().strip()]
        analyze_mode = analyze_var.get()
        if analyze_mode:
            cmd.append("--analyze-only")

        log.config(state="normal")
        log.delete("1.0", tk.END)
        log.config(state="disabled")
        run_btn.config(state="disabled")
        btn_label_var.set(_t("processing"))

        def done(ok):
            run_btn.config(state="normal")
            btn_label_var.set(_t("btn_summarize"))
            if not ok:
                messagebox.showerror(_t("msg_error"), _t("msg_error_body"))
                return
            if analyze_mode and _load_analysis_into_manual(log):
                nb.select(MANUAL_TAB["frame"])
                messagebox.showinfo(_t("msg_done"), _t("msg_analyze_done"))
            else:
                messagebox.showinfo(
                    _t("msg_done"),
                    _t("msg_summ_done").format(folder=outdir_var.get()))

        run_script(cmd, log, done)

    run_btn.config(command=on_run)
    return frame


# ── 탭 2: 수동 하이라이트 ──────────────────────────────────────────────────────

def build_manual_tab(nb):
    frame = ttk.Frame(nb, padding=4)
    nb.add(frame, text=_t("tab_manual"))
    reg("tab", (nb, frame), "tab_manual")
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(8, weight=1)

    pad = {"padx": 12, "pady": 4}

    heading = ttk.Label(frame, text=_t("man_heading"), font=("Segoe UI", 13, "bold"))
    reg("text", heading, "man_heading")
    heading.grid(row=0, column=0, columnspan=3, sticky="w", padx=12, pady=(10, 8))

    video_var = tk.StringVar()
    outdir_var = tk.StringVar(value=os.path.join(SCRIPT_DIR, "output"))
    name_var = tk.StringVar()

    # 영상 파일
    _label(frame, "man_video", row=1, column=0, sticky="w", **pad)
    ttk.Entry(frame, textvariable=video_var, width=52).grid(
        row=1, column=1, sticky="ew", padx=(0, 4), pady=4)
    browse_vid = ttk.Button(
        frame, text=_t("browse"),
        command=lambda: video_var.set(
            filedialog.askopenfilename(
                title=_t("dlg_man_video"),
                filetypes=[(_t("ft_video"), "*.mp4 *.mov *.mkv *.avi *.webm"),
                           (_t("ft_all"), "*.*")]) or video_var.get()))
    reg("text", browse_vid, "browse")
    browse_vid.grid(row=1, column=2, padx=(0, 12), pady=4)

    # 출력 폴더
    _label(frame, "outdir", row=2, column=0, sticky="w", **pad)
    ttk.Entry(frame, textvariable=outdir_var, width=52).grid(
        row=2, column=1, sticky="ew", padx=(0, 4), pady=4)
    browse_out = ttk.Button(
        frame, text=_t("browse"),
        command=lambda: outdir_var.set(
            filedialog.askdirectory(title=_t("dlg_outdir")) or outdir_var.get()))
    reg("text", browse_out, "browse")
    browse_out.grid(row=2, column=2, padx=(0, 12), pady=4)

    # 출력 이름 (선택)
    _label(frame, "man_name", row=3, column=0, sticky="w", **pad)
    ttk.Entry(frame, textvariable=name_var, width=52).grid(
        row=3, column=1, columnspan=2, sticky="ew", padx=(0, 12), pady=4)

    # 하이라이트 시간대 입력
    _label(frame, "man_ranges", row=4, column=0, sticky="nw", padx=12, pady=(8, 0))
    ranges_hint = ttk.Label(frame, text=_t("man_ranges_hint"), foreground="#9ca3af")
    reg("text", ranges_hint, "man_ranges_hint")
    ranges_hint.grid(row=4, column=1, columnspan=2, sticky="w", padx=(0, 12), pady=(8, 0))

    ranges_text = tk.Text(frame, height=6, bg="#3c3c3c", fg="#e0e0e0",
                          insertbackground="#e0e0e0", relief="flat",
                          font=("Consolas", 10), wrap="none")
    ranges_text.insert("1.0", _t("man_ranges_example"))
    ranges_text.grid(row=5, column=0, columnspan=3, sticky="ew", padx=12, pady=(2, 6))

    # 옵션
    opt = ttk.LabelFrame(frame, text=_t("options"), padding=8)
    reg("text", opt, "options")
    opt.grid(row=6, column=0, columnspan=3, sticky="ew", padx=12, pady=6)
    opt.columnconfigure(1, weight=1)
    opt.columnconfigure(3, weight=1)

    _label(opt, "transition", row=0, column=0, sticky="w", padx=(8, 4), pady=3)
    trans_combo = ttk.Combobox(opt, values=_t("transition_values"), width=14, state="readonly")
    trans_combo.current(1)  # black
    reg("combo", (trans_combo, "transition_values"), "transition_values")
    trans_combo.grid(row=0, column=1, sticky="w", pady=3)
    _label(opt, "sfx", row=0, column=2, sticky="w", padx=(16, 4), pady=3)
    sfx_combo = ttk.Combobox(opt, values=_t("sfx_values"), width=14, state="readonly")
    sfx_combo.current(1)  # whoosh
    reg("combo", (sfx_combo, "sfx_values"), "sfx_values")
    sfx_combo.grid(row=0, column=3, sticky="w", pady=3)

    subs_var = tk.BooleanVar(value=False)
    chk_subs = ttk.Checkbutton(opt, text=_t("man_subtitles"), variable=subs_var)
    reg("text", chk_subs, "man_subtitles")
    chk_subs.grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 3))

    lang_var = tk.StringVar(value="ko")
    _label(opt, "model", row=2, column=0, sticky="w", padx=(8, 4), pady=3)
    model_combo = ttk.Combobox(opt, values=_t("model_values"), width=16, state="readonly")
    model_combo.current(2)  # small
    reg("combo", (model_combo, "model_values"), "model_values")
    model_combo.grid(row=2, column=1, sticky="w", pady=3)
    _label(opt, "language", row=2, column=2, sticky="w", padx=(16, 4), pady=3)
    ttk.Entry(opt, textvariable=lang_var, width=6).grid(row=2, column=3, sticky="w", pady=3)

    # 하이라이트 소제목 위치 ( '시작-끝 | 소제목' 으로 입력한 소제목이 뜨는 자리 )
    _label(opt, "label_position", row=3, column=0, sticky="w", padx=(8, 4), pady=(6, 3))
    labelpos_combo = ttk.Combobox(opt, values=_t("wm_pos_values"), width=10, state="readonly")
    labelpos_combo.current(1)  # tr = 우상단
    reg("combo", (labelpos_combo, "wm_pos_values"), "wm_pos_values")
    labelpos_combo.grid(row=3, column=1, sticky="w", pady=(6, 3))
    _label(opt, "label_position_hint", row=3, column=2, columnspan=2, sticky="w", padx=(8, 4), pady=(6, 3))

    # 실행 버튼
    btn_label_var = tk.StringVar(value=_t("btn_manual"))
    reg("var", btn_label_var, "btn_manual")
    run_btn = ttk.Button(frame, textvariable=btn_label_var, style="Accent.TButton")
    run_btn.grid(row=7, column=0, columnspan=3, padx=12, pady=8, sticky="ew")

    # 로그
    log = make_log(frame)
    log.grid(row=8, column=0, columnspan=3, sticky="nsew", padx=12, pady=(0, 12))

    def on_run():
        video = video_var.get().strip()
        ranges = ranges_text.get("1.0", tk.END).strip()
        if not video:
            messagebox.showwarning(_t("msg_input_error"), _t("msg_need_man_video"))
            return
        # 주석(#)·빈 줄을 제외하고 실제 입력이 있는지 확인
        has_range = any(
            ln.strip() and not ln.strip().startswith("#")
            for ln in ranges.splitlines())
        if not has_range:
            messagebox.showwarning(_t("msg_input_error"), _t("msg_need_ranges"))
            return

        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "manual_highlight.py"),
            video,
            "--ranges", ranges,
            "--output-dir", outdir_var.get(),
            "--transition-style", TRANS_CODES[max(trans_combo.current(), 0)],
            "--sfx", SFX_CODES[max(sfx_combo.current(), 0)],
        ]
        if name_var.get().strip():
            cmd += ["--name", name_var.get().strip()]
        cmd += ["--label-pos", WMPOS_CODES[max(labelpos_combo.current(), 0)]]
        if subs_var.get():
            cmd += ["--subtitles",
                    "--model", MODEL_CODES[max(model_combo.current(), 0)],
                    "--lang", lang_var.get()]

        log.config(state="normal")
        log.delete("1.0", tk.END)
        log.config(state="disabled")
        run_btn.config(state="disabled")
        btn_label_var.set(_t("processing"))

        def done(ok):
            run_btn.config(state="normal")
            btn_label_var.set(_t("btn_manual"))
            if ok:
                messagebox.showinfo(
                    _t("msg_done"),
                    _t("msg_man_done").format(folder=outdir_var.get()))
            else:
                messagebox.showerror(_t("msg_error"), _t("msg_error_body"))

        run_script(cmd, log, done)

    run_btn.config(command=on_run)

    # 요약 탭의 분석 전용 모드가 결과를 넘겨줄 수 있게 위젯 참조를 공유
    MANUAL_TAB.update(frame=frame, video_var=video_var, ranges_text=ranges_text)
    return frame


# ── 탭 3: 쇼츠 만들기 ─────────────────────────────────────────────────────────

def build_shorts_tab(nb):
    frame = ttk.Frame(nb, padding=4)
    nb.add(frame, text=_t("tab_shorts"))
    reg("tab", (nb, frame), "tab_shorts")
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(8, weight=1)

    pad = {"padx": 12, "pady": 4}

    heading = ttk.Label(frame, text=_t("shorts_heading"), font=("Segoe UI", 13, "bold"))
    reg("text", heading, "shorts_heading")
    heading.grid(row=0, column=0, columnspan=3, sticky="w", padx=12, pady=(10, 8))

    video_var = tk.StringVar()
    outdir_var = tk.StringVar(value=os.path.join(SCRIPT_DIR, "output"))
    name_var = tk.StringVar()

    # 영상 파일
    _label(frame, "man_video", row=1, column=0, sticky="w", **pad)
    ttk.Entry(frame, textvariable=video_var, width=52).grid(
        row=1, column=1, sticky="ew", padx=(0, 4), pady=4)
    browse_vid = ttk.Button(
        frame, text=_t("browse"),
        command=lambda: video_var.set(
            filedialog.askopenfilename(
                title=_t("dlg_man_video"),
                filetypes=[(_t("ft_video"), "*.mp4 *.mov *.mkv *.avi *.webm"),
                           (_t("ft_all"), "*.*")]) or video_var.get()))
    reg("text", browse_vid, "browse")
    browse_vid.grid(row=1, column=2, padx=(0, 12), pady=4)

    # 출력 폴더
    _label(frame, "outdir", row=2, column=0, sticky="w", **pad)
    ttk.Entry(frame, textvariable=outdir_var, width=52).grid(
        row=2, column=1, sticky="ew", padx=(0, 4), pady=4)
    browse_out = ttk.Button(
        frame, text=_t("browse"),
        command=lambda: outdir_var.set(
            filedialog.askdirectory(title=_t("dlg_outdir")) or outdir_var.get()))
    reg("text", browse_out, "browse")
    browse_out.grid(row=2, column=2, padx=(0, 12), pady=4)

    # 출력 이름 (선택)
    _label(frame, "man_name", row=3, column=0, sticky="w", **pad)
    ttk.Entry(frame, textvariable=name_var, width=52).grid(
        row=3, column=1, columnspan=2, sticky="ew", padx=(0, 12), pady=4)

    # 쇼츠 구간 입력
    _label(frame, "shorts_ranges", row=4, column=0, sticky="nw", padx=12, pady=(8, 0))
    ranges_hint = ttk.Label(frame, text=_t("shorts_ranges_hint"), foreground="#9ca3af")
    reg("text", ranges_hint, "shorts_ranges_hint")
    ranges_hint.grid(row=4, column=1, columnspan=2, sticky="w", padx=(0, 12), pady=(8, 0))

    ranges_text = tk.Text(frame, height=4, bg="#3c3c3c", fg="#e0e0e0",
                          insertbackground="#e0e0e0", relief="flat",
                          font=("Consolas", 10), wrap="none")
    ranges_text.insert("1.0", _t("shorts_ranges_example"))
    ranges_text.grid(row=5, column=0, columnspan=3, sticky="ew", padx=12, pady=(2, 6))

    # 옵션
    opt = ttk.LabelFrame(frame, text=_t("options"), padding=8)
    reg("text", opt, "options")
    opt.grid(row=6, column=0, columnspan=3, sticky="ew", padx=12, pady=6)
    opt.columnconfigure(1, weight=1)
    opt.columnconfigure(3, weight=1)

    _label(opt, "shorts_mode", row=0, column=0, sticky="w", padx=(8, 4), pady=3)
    mode_combo = ttk.Combobox(opt, values=_t("shorts_mode_values"), width=18, state="readonly")
    mode_combo.current(0)  # center
    reg("combo", (mode_combo, "shorts_mode_values"), "shorts_mode_values")
    mode_combo.grid(row=0, column=1, sticky="w", pady=3)
    fontsize_var = tk.StringVar(value="54")
    _label(opt, "shorts_font_size", row=0, column=2, sticky="w", padx=(16, 4), pady=3)
    ttk.Entry(opt, textvariable=fontsize_var, width=6).grid(row=0, column=3, sticky="w", pady=3)
    _label(opt, "shorts_mode_hint", row=1, column=0, columnspan=4, sticky="w", padx=(8, 4), pady=(0, 3))

    _label(opt, "shorts_sub_pos", row=2, column=0, sticky="w", padx=(8, 4), pady=3)
    subpos_combo = ttk.Combobox(opt, values=_t("shorts_sub_pos_values"), width=12, state="readonly")
    subpos_combo.current(0)  # bottom
    reg("combo", (subpos_combo, "shorts_sub_pos_values"), "shorts_sub_pos_values")
    subpos_combo.grid(row=2, column=1, sticky="w", pady=3)

    subs_var = tk.BooleanVar(value=False)
    chk_subs = ttk.Checkbutton(opt, text=_t("shorts_subtitles"), variable=subs_var)
    reg("text", chk_subs, "shorts_subtitles")
    chk_subs.grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 3))

    lang_var = tk.StringVar(value="ko")
    _label(opt, "model", row=4, column=0, sticky="w", padx=(8, 4), pady=3)
    model_combo = ttk.Combobox(opt, values=_t("model_values"), width=16, state="readonly")
    model_combo.current(2)  # small
    reg("combo", (model_combo, "model_values"), "model_values")
    model_combo.grid(row=4, column=1, sticky="w", pady=3)
    _label(opt, "language", row=4, column=2, sticky="w", padx=(16, 4), pady=3)
    ttk.Entry(opt, textvariable=lang_var, width=6).grid(row=4, column=3, sticky="w", pady=3)

    # 실행 버튼
    btn_label_var = tk.StringVar(value=_t("btn_shorts"))
    reg("var", btn_label_var, "btn_shorts")
    run_btn = ttk.Button(frame, textvariable=btn_label_var, style="Accent.TButton")
    run_btn.grid(row=7, column=0, columnspan=3, padx=12, pady=8, sticky="ew")

    # 로그
    log = make_log(frame)
    log.grid(row=8, column=0, columnspan=3, sticky="nsew", padx=12, pady=(0, 12))

    def on_run():
        video = video_var.get().strip()
        ranges = ranges_text.get("1.0", tk.END).strip()
        if not video:
            messagebox.showwarning(_t("msg_input_error"), _t("msg_need_man_video"))
            return
        has_range = any(
            ln.strip() and not ln.strip().startswith("#")
            for ln in ranges.splitlines())
        if not has_range:
            messagebox.showwarning(_t("msg_input_error"), _t("msg_need_ranges"))
            return

        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "shorts.py"),
            video,
            "--ranges", ranges,
            "--output-dir", outdir_var.get(),
            "--mode", SHORTS_MODE_CODES[max(mode_combo.current(), 0)],
        ]
        if name_var.get().strip():
            cmd += ["--name", name_var.get().strip()]
        if fontsize_var.get().strip():
            cmd += ["--font-size", fontsize_var.get().strip()]
        cmd += ["--sub-pos", SHORTS_SUBPOS_CODES[max(subpos_combo.current(), 0)]]
        if subs_var.get():
            cmd += ["--subtitles",
                    "--model", MODEL_CODES[max(model_combo.current(), 0)],
                    "--lang", lang_var.get()]

        log.config(state="normal")
        log.delete("1.0", tk.END)
        log.config(state="disabled")
        run_btn.config(state="disabled")
        btn_label_var.set(_t("processing"))

        def done(ok):
            run_btn.config(state="normal")
            btn_label_var.set(_t("btn_shorts"))
            if ok:
                messagebox.showinfo(
                    _t("msg_done"),
                    _t("msg_shorts_done").format(folder=outdir_var.get()))
            else:
                messagebox.showerror(_t("msg_error"), _t("msg_error_body"))

        run_script(cmd, log, done)

    run_btn.config(command=on_run)
    return frame


# ── 탭 4: 완성 영상 만들기 ─────────────────────────────────────────────────────

def build_finalize_tab(nb):
    frame = ttk.Frame(nb)
    nb.add(frame, text=_t("tab_finalize"))
    reg("tab", (nb, frame), "tab_finalize")
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(9, weight=1)

    pad = {"padx": 12, "pady": 4}

    def vid_types():
        return [(_t("ft_video"), "*.mp4 *.mov *.mkv *.avi *.webm"), (_t("ft_all"), "*.*")]

    def audio_types():
        return [(_t("ft_audio"), "*.mp3 *.m4a *.aac *.wav *.flac *.ogg *.opus"),
                (_t("ft_all"), "*.*")]

    def browse_row(row, label_key, var, dlg_key, filetypes_fn, clearable=False):
        _label(frame, label_key, row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=var, width=52).grid(
            row=row, column=1, sticky="ew", padx=(0, 4), pady=4)
        btns = ttk.Frame(frame)
        btns.grid(row=row, column=2, padx=(0, 12), pady=4, sticky="w")
        b = ttk.Button(btns, text=_t("browse"),
                       command=lambda: var.set(
                           filedialog.askopenfilename(title=_t(dlg_key),
                                                      filetypes=filetypes_fn())
                           or var.get()))
        reg("text", b, "browse")
        b.pack(side="left")
        if clearable:
            ttk.Button(btns, text="✕", width=3,
                       command=lambda: var.set("")).pack(side="left", padx=(4, 0))

    video_var = tk.StringVar()
    srt_var = tk.StringVar()
    thumb_var = tk.StringVar()
    intro_video_var = tk.StringVar()
    outro_video_var = tk.StringVar()
    bgm_var = tk.StringVar()
    out_var = tk.StringVar()

    browse_row(0, "video", video_var, "dlg_video", vid_types)
    browse_row(1, "srt", srt_var, "dlg_srt",
               lambda: [(_t("ft_subtitle"), "*.srt *.ass"), (_t("ft_all"), "*.*")])
    browse_row(2, "thumb", thumb_var, "dlg_thumb",
               lambda: [(_t("ft_image"), "*.jpg *.jpeg *.png *.webp *.bmp"), (_t("ft_all"), "*.*")])
    browse_row(3, "intro_video", intro_video_var, "dlg_intro", vid_types, clearable=True)
    browse_row(4, "outro_video", outro_video_var, "dlg_outro", vid_types, clearable=True)
    browse_row(5, "bgm", bgm_var, "dlg_bgm", audio_types, clearable=True)

    # 영상 선택 시 같은 폴더/이름의 srt·출력경로 자동 추정
    def autofill(*_):
        v = video_var.get().strip()
        if not v:
            return
        base, _ext = os.path.splitext(v)
        cand_srt = base + ".srt"
        if not srt_var.get().strip() and os.path.isfile(cand_srt):
            srt_var.set(cand_srt)
        if not out_var.get().strip():
            out_var.set(base + "_final.mp4")
    video_var.trace_add("write", autofill)

    # 출력 파일
    _label(frame, "outfile", row=6, column=0, sticky="w", **pad)
    ttk.Entry(frame, textvariable=out_var, width=52).grid(
        row=6, column=1, sticky="ew", padx=(0, 4), pady=4)
    save_btn = ttk.Button(
        frame, text=_t("save_as"),
        command=lambda: out_var.set(
            filedialog.asksaveasfilename(
                title=_t("dlg_save"), defaultextension=".mp4",
                filetypes=[(_t("ft_mp4"), "*.mp4")]) or out_var.get()))
    reg("text", save_btn, "save_as")
    save_btn.grid(row=6, column=2, padx=(0, 12), pady=4)

    # 옵션
    opt = ttk.LabelFrame(frame, text=_t("options"), padding=8)
    reg("text", opt, "options")
    opt.grid(row=7, column=0, columnspan=3, sticky="ew", padx=12, pady=6)
    opt.columnconfigure(1, weight=1)
    opt.columnconfigure(3, weight=1)

    intro_var = tk.BooleanVar(value=True)
    cover_var = tk.BooleanVar(value=True)
    burn_var = tk.BooleanVar(value=True)
    intro_sec_var = tk.StringVar(value="2.5")
    font_size_var = tk.StringVar(value="24")
    bgm_volume_var = tk.StringVar(value="0.25")

    chk_intro = ttk.Checkbutton(opt, text=_t("opt_intro"), variable=intro_var)
    reg("text", chk_intro, "opt_intro")
    chk_intro.grid(row=0, column=0, sticky="w", padx=8, pady=3)
    _label(opt, "intro_sec", row=0, column=2, sticky="w", padx=(16, 4), pady=3)
    ttk.Entry(opt, textvariable=intro_sec_var, width=6).grid(row=0, column=3, sticky="w", pady=3)

    chk_cover = ttk.Checkbutton(opt, text=_t("opt_cover"), variable=cover_var)
    reg("text", chk_cover, "opt_cover")
    chk_cover.grid(row=1, column=0, sticky="w", padx=8, pady=3)
    _label(opt, "font_size", row=1, column=2, sticky="w", padx=(16, 4), pady=3)
    ttk.Entry(opt, textvariable=font_size_var, width=6).grid(row=1, column=3, sticky="w", pady=3)

    chk_burn = ttk.Checkbutton(opt, text=_t("opt_burn"), variable=burn_var)
    reg("text", chk_burn, "opt_burn")
    chk_burn.grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=3)
    _label(opt, "bgm_volume", row=2, column=2, sticky="w", padx=(16, 4), pady=3)
    ttk.Entry(opt, textvariable=bgm_volume_var, width=6).grid(row=2, column=3, sticky="w", pady=3)

    # 채널 마크(워터마크) 이미지 (선택) — 본영상에만 새겨진다
    watermark_var = tk.StringVar(value="")
    _label(opt, "watermark", row=3, column=0, sticky="w", padx=(8, 4), pady=(6, 3))
    ttk.Entry(opt, textvariable=watermark_var, width=28).grid(
        row=3, column=1, columnspan=2, sticky="ew", pady=(6, 3))
    wm_btns = ttk.Frame(opt)
    wm_btns.grid(row=3, column=3, sticky="w", padx=(8, 4), pady=(6, 3))
    browse_wm = ttk.Button(
        wm_btns, text=_t("browse"),
        command=lambda: watermark_var.set(
            filedialog.askopenfilename(
                title=_t("dlg_watermark"),
                filetypes=[(_t("ft_image"), "*.png *.jpg *.jpeg *.webp *.bmp"),
                           (_t("ft_all"), "*.*")]) or watermark_var.get()))
    reg("text", browse_wm, "browse")
    browse_wm.pack(side="left")
    ttk.Button(wm_btns, text="✕", width=3,
               command=lambda: watermark_var.set("")).pack(side="left", padx=(4, 0))
    _label(opt, "wm_position", row=4, column=0, sticky="w", padx=(8, 4), pady=(0, 3))
    wmpos_combo = ttk.Combobox(opt, values=_t("wm_pos_values"), width=10, state="readonly")
    wmpos_combo.current(1)  # tr = 우상단
    reg("combo", (wmpos_combo, "wm_pos_values"), "wm_pos_values")
    wmpos_combo.grid(row=4, column=1, sticky="w", pady=(0, 3))
    _label(opt, "watermark_hint", row=4, column=2, columnspan=2, sticky="w", padx=(8, 4), pady=(0, 3))

    _label(opt, "wm_colorkey", row=5, column=0, sticky="w", padx=(8, 4), pady=(0, 3))
    wmkey_combo = ttk.Combobox(opt, values=_t("wm_colorkey_values"), width=10, state="readonly")
    wmkey_combo.current(0)  # 없음
    reg("combo", (wmkey_combo, "wm_colorkey_values"), "wm_colorkey_values")
    wmkey_combo.grid(row=5, column=1, sticky="w", pady=(0, 3))
    _label(opt, "wm_colorkey_hint", row=5, column=2, columnspan=2, sticky="w", padx=(8, 4), pady=(0, 3))

    # AI 자동 키워드 (Gemini) — 자막을 분석해 구간별 키워드를 마크 아래 표시
    autolabels_var = tk.BooleanVar(value=False)
    chk_labels = ttk.Checkbutton(opt, text=_t("auto_labels"), variable=autolabels_var)
    reg("text", chk_labels, "auto_labels")
    chk_labels.grid(row=6, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 0))
    _label(opt, "auto_labels_hint", row=6, column=2, columnspan=2, sticky="w", padx=(8, 4), pady=(8, 0))
    _label(opt, "gemini_key", row=7, column=0, sticky="w", padx=(8, 4), pady=(0, 3))
    gemini_key_var = tk.StringVar(value=load_gemini_key())
    ttk.Entry(opt, textvariable=gemini_key_var, width=28, show="•").grid(
        row=7, column=1, columnspan=2, sticky="ew", pady=(0, 3))
    _label(opt, "gemini_key_hint", row=7, column=3, sticky="w", padx=(8, 4), pady=(0, 3))

    # 실행 버튼
    btn_label_var = tk.StringVar(value=_t("btn_finalize"))
    reg("var", btn_label_var, "btn_finalize")
    run_btn = ttk.Button(frame, textvariable=btn_label_var, style="Accent.TButton")
    run_btn.grid(row=8, column=0, columnspan=3, padx=12, pady=8, sticky="ew")

    # 로그
    log = make_log(frame)
    log.grid(row=9, column=0, columnspan=3, sticky="nsew", padx=12, pady=(0, 12))

    def on_run():
        video = video_var.get().strip()
        srt = srt_var.get().strip()
        thumb = thumb_var.get().strip()
        out = out_var.get().strip()
        if not video:
            messagebox.showwarning(_t("msg_input_error"), _t("msg_need_video"))
            return
        if burn_var.get() and not srt:
            messagebox.showwarning(_t("msg_input_error"), _t("msg_need_srt"))
            return
        if (intro_var.get() or cover_var.get()) and not thumb:
            messagebox.showwarning(_t("msg_input_error"), _t("msg_need_thumb"))
            return
        if not out:
            messagebox.showwarning(_t("msg_input_error"), _t("msg_need_out"))
            return

        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "finalize.py"),
            video, srt or "-", thumb or "-",
            "-o", out,
            "--intro-sec", intro_sec_var.get(),
            "--font-size", font_size_var.get(),
        ]
        if not intro_var.get():
            cmd += ["--no-intro"]
        if not cover_var.get():
            cmd += ["--no-cover"]
        if not burn_var.get():
            cmd += ["--no-subs"]
        if intro_video_var.get().strip():
            cmd += ["--intro-video", intro_video_var.get().strip()]
        if outro_video_var.get().strip():
            cmd += ["--outro-video", outro_video_var.get().strip()]
        if bgm_var.get().strip():
            cmd += ["--bgm", bgm_var.get().strip(),
                    "--bgm-volume", bgm_volume_var.get().strip() or "0.25"]
        wm_pos = WMPOS_CODES[max(wmpos_combo.current(), 0)]
        if watermark_var.get().strip():
            cmd += ["--watermark", watermark_var.get().strip(), "--wm-pos", wm_pos]
            wmkey = WMKEY_CODES[max(wmkey_combo.current(), 0)]
            if wmkey:
                cmd += ["--wm-colorkey", wmkey]

        # AI 자동 키워드 (Gemini)
        gkey = gemini_key_var.get().strip()
        if autolabels_var.get():
            if not gkey:
                messagebox.showwarning(_t("msg_input_error"), _t("gemini_key"))
                return
            if not srt:
                messagebox.showwarning(_t("msg_input_error"), _t("msg_need_srt_labels"))
                return
            save_gemini_key(gkey)
            cmd += ["--auto-labels", "--gemini-key", gkey]
            # 라벨 위치(마크 아래)를 위해 위치를 항상 전달 (마크가 없어도)
            if "--wm-pos" not in cmd:
                cmd += ["--wm-pos", wm_pos]

        log.config(state="normal")
        log.delete("1.0", tk.END)
        log.config(state="disabled")
        run_btn.config(state="disabled")
        btn_label_var.set(_t("processing"))

        def done(ok):
            run_btn.config(state="normal")
            btn_label_var.set(_t("btn_finalize"))
            if ok:
                messagebox.showinfo(_t("msg_done"),
                                    _t("msg_final_done").format(path=out))
            else:
                messagebox.showerror(_t("msg_error"), _t("msg_error_body"))

        run_script(cmd, log, done)

    run_btn.config(command=on_run)
    return frame


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.title(_t("win_title"))
    reg("title", root, "win_title")
    root.geometry("720x800")
    root.minsize(640, 660)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure("TFrame", background="#2d2d2d")
    style.configure("TLabel", background="#2d2d2d", foreground="#e0e0e0")
    style.configure("TLabelframe", background="#2d2d2d", foreground="#e0e0e0")
    style.configure("TLabelframe.Label", background="#2d2d2d", foreground="#aaaaaa")
    style.configure("TCheckbutton", background="#2d2d2d", foreground="#e0e0e0")
    style.configure("TEntry", fieldbackground="#3c3c3c", foreground="#e0e0e0", insertcolor="#e0e0e0")
    style.configure("TCombobox", fieldbackground="#3c3c3c", foreground="#e0e0e0")
    style.configure("TNotebook", background="#2d2d2d", tabmargins=[2, 5, 2, 0])
    style.configure("TNotebook.Tab", background="#3c3c3c", foreground="#cccccc",
                    padding=[10, 4], font=("Segoe UI", 10))
    style.map("TNotebook.Tab",
              background=[("selected", "#1e1e1e")],
              foreground=[("selected", "#ffffff")])
    style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"),
                    background="#0078d4", foreground="#ffffff", padding=8)
    style.map("Accent.TButton",
              background=[("active", "#005fa3"), ("disabled", "#555555")],
              foreground=[("disabled", "#888888")])
    style.configure("TButton", background="#3c3c3c", foreground="#e0e0e0")
    style.map("TButton", background=[("active", "#505050")])
    style.configure("Lang.TButton", background="#3c3c3c", foreground="#e0e0e0",
                    font=("Segoe UI", 9))
    root.configure(bg="#2d2d2d")

    # 상단 바: 언어 전환 버튼 / top bar: language toggle
    top = ttk.Frame(root)
    top.pack(fill="x")

    def toggle_lang():
        STATE["lang"] = "en" if STATE["lang"] == "ko" else "ko"
        lang_btn.config(text=_t("lang_button"))
        apply_language()

    lang_btn = ttk.Button(top, text=_t("lang_button"), style="Lang.TButton",
                          command=toggle_lang)
    lang_btn.pack(side="right", padx=8, pady=(6, 0))

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)

    build_summarizer_tab(nb)
    build_manual_tab(nb)
    build_shorts_tab(nb)
    build_finalize_tab(nb)

    root.mainloop()


if __name__ == "__main__":
    main()

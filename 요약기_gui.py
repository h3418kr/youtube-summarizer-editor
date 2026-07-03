"""GUI launcher for summarizer.py (YouTube 요약기 단독판)

두 개의 탭:
  1. 영상 요약    — YouTube URL → 요약 영상 + 자막(SRT)
  2. 완성 영상 만들기 — 영상 + 자막 + 썸네일 → 완성 mp4 (finalize.py)
"""
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


# ── 탭 1: 영상 요약 ────────────────────────────────────────────────────────────

def build_summarizer_tab(nb):
    frame = ttk.Frame(nb, padding=4)
    nb.add(frame, text="  영상 요약  ")
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(5, weight=1)

    pad = {"padx": 12, "pady": 4}

    # 제목
    ttk.Label(frame, text="YouTube 영상 요약기",
              font=("Segoe UI", 13, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", padx=12, pady=(10, 8))

    # URL
    ttk.Label(frame, text="YouTube URL").grid(row=1, column=0, sticky="w", **pad)
    url_var = tk.StringVar()
    ttk.Entry(frame, textvariable=url_var, width=60).grid(
        row=1, column=1, columnspan=2, sticky="ew", padx=(0, 12), pady=4)

    # 출력 폴더
    ttk.Label(frame, text="출력 폴더").grid(row=2, column=0, sticky="w", **pad)
    outdir_var = tk.StringVar(value=os.path.join(SCRIPT_DIR, "output"))
    ttk.Entry(frame, textvariable=outdir_var, width=52).grid(
        row=2, column=1, sticky="ew", padx=(0, 4), pady=4)
    ttk.Button(frame, text="찾아보기",
               command=lambda: outdir_var.set(
                   filedialog.askdirectory(title="출력 폴더 선택") or outdir_var.get()
               )).grid(row=2, column=2, padx=(0, 12), pady=4)

    # 옵션
    opt = ttk.LabelFrame(frame, text="옵션", padding=8)
    opt.grid(row=3, column=0, columnspan=3, sticky="ew", padx=12, pady=6)
    opt.columnconfigure(1, weight=1)
    opt.columnconfigure(3, weight=1)

    target_var = tk.StringVar(value="10")
    model_var = tk.StringVar(value="small (권장)")
    lang_var = tk.StringVar(value="ko")
    before_var = tk.StringVar(value="5")
    after_var = tk.StringVar(value="20")
    quality_var = tk.StringVar(value="720")
    transition_var = tk.StringVar(value="암전 (기본)")
    sfx_var = tk.StringVar(value="휙 (기본)")
    bridge_var = tk.StringVar(value="8")

    ttk.Label(opt, text="목표 길이 (분)").grid(row=0, column=0, sticky="w", padx=(8, 4), pady=3)
    ttk.Entry(opt, textvariable=target_var, width=6).grid(row=0, column=1, sticky="w", pady=3)
    ttk.Label(opt, text="Whisper 모델 (자막 품질)").grid(row=0, column=2, sticky="w", padx=(16, 4), pady=3)
    ttk.Combobox(opt, textvariable=model_var,
                 values=["tiny (빠름)", "base", "small (권장)", "medium (정확)", "large (최고)"],
                 width=14, state="readonly").grid(row=0, column=3, sticky="w", pady=3)

    ttk.Label(opt, text="언어").grid(row=1, column=0, sticky="w", padx=(8, 4), pady=3)
    ttk.Entry(opt, textvariable=lang_var, width=6).grid(row=1, column=1, sticky="w", pady=3)
    ttk.Label(opt, text="화질").grid(row=1, column=2, sticky="w", padx=(16, 4), pady=3)
    ttk.Combobox(opt, textvariable=quality_var,
                 values=["360", "480", "720", "1080"],
                 width=6, state="readonly").grid(row=1, column=3, sticky="w", pady=3)

    ttk.Label(opt, text="피크 앞 확장 (초)").grid(row=2, column=0, sticky="w", padx=(8, 4), pady=3)
    ttk.Entry(opt, textvariable=before_var, width=6).grid(row=2, column=1, sticky="w", pady=3)
    ttk.Label(opt, text="피크 뒤 확장 (초)").grid(row=2, column=2, sticky="w", padx=(16, 4), pady=3)
    ttk.Entry(opt, textvariable=after_var, width=6).grid(row=2, column=3, sticky="w", pady=3)

    ttk.Label(opt, text="같은 장면 묶기 기준 (초)").grid(row=3, column=0, sticky="w", padx=(8, 4), pady=3)
    ttk.Entry(opt, textvariable=bridge_var, width=6).grid(row=3, column=1, sticky="w", pady=3)
    ttk.Label(opt, text="(시간차가 이보다 짧으면 한 장면으로 이어붙임)").grid(
        row=3, column=2, columnspan=2, sticky="w", padx=(16, 4), pady=3)

    ttk.Label(opt, text="화면 전환").grid(row=4, column=0, sticky="w", padx=(8, 4), pady=(6, 3))
    ttk.Combobox(opt, textvariable=transition_var,
                 values=["없음", "암전 (기본)", "화이트 플래시"],
                 width=14, state="readonly").grid(row=4, column=1, sticky="w", pady=(6, 3))
    ttk.Label(opt, text="전환 효과음").grid(row=4, column=2, sticky="w", padx=(16, 4), pady=(6, 3))
    ttk.Combobox(opt, textvariable=sfx_var,
                 values=["없음", "휙 (기본)", "스와이프", "삑", "팝", "임팩트"],
                 width=14, state="readonly").grid(row=4, column=3, sticky="w", pady=(6, 3))
    ttk.Label(opt, text="(서로 다른 하이라이트 사이에 적용됩니다)").grid(
        row=5, column=0, columnspan=4, sticky="w", padx=(8, 4), pady=(0, 3))

    # 실행 버튼
    btn_label_var = tk.StringVar(value="다운로드 & 요약")
    run_btn = ttk.Button(frame, textvariable=btn_label_var, style="Accent.TButton")
    run_btn.grid(row=4, column=0, columnspan=3, padx=12, pady=8, sticky="new")

    # 로그
    log = make_log(frame)
    log.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=12, pady=(0, 12))

    def on_run():
        url = url_var.get().strip()
        if not url:
            messagebox.showwarning("입력 오류", "YouTube URL을 입력하세요.")
            return

        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "summarizer.py"),
            url,
            "--target-min", target_var.get(),
            "--model", model_var.get().split(" ")[0],
            "--lang", lang_var.get(),
            "--expand-before", before_var.get(),
            "--expand-after", after_var.get(),
            "--output-dir", outdir_var.get(),
            "--max-height", quality_var.get(),
            "--bridge-gap", bridge_var.get(),
        ]
        trans_map = {"없음": "none", "암전 (기본)": "black", "화이트 플래시": "white"}
        sfx_map = {"없음": "none", "휙 (기본)": "whoosh", "스와이프": "swoosh",
                   "삑": "beep", "팝": "pop", "임팩트": "impact"}
        cmd += ["--transition-style", trans_map.get(transition_var.get(), "black"),
                "--sfx", sfx_map.get(sfx_var.get(), "whoosh")]

        log.config(state="normal")
        log.delete("1.0", tk.END)
        log.config(state="disabled")
        run_btn.config(state="disabled")
        btn_label_var.set("처리 중...")

        def done(ok):
            run_btn.config(state="normal")
            btn_label_var.set("다운로드 & 요약")
            if ok:
                messagebox.showinfo("완료",
                    f"저장 완료!\n\n"
                    f"폴더: {outdir_var.get()}\n\n"
                    f"• _summary.mp4  — 요약 영상\n"
                    f"• _summary.srt  — 자막 파일 (편집 후 영상에 적용)"
                )
            else:
                messagebox.showerror("오류", "처리 중 오류가 발생했습니다.\n로그를 확인하세요.")

        run_script(cmd, log, done)

    run_btn.config(command=on_run)
    return frame


# ── 탭 2: 완성 영상 만들기 ─────────────────────────────────────────────────────

def build_finalize_tab(nb):
    frame = ttk.Frame(nb)
    nb.add(frame, text="  완성 영상 만들기  ")
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(6, weight=1)

    pad = {"padx": 12, "pady": 4}

    def browse_row(row, label, var, title, filetypes):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=var, width=52).grid(
            row=row, column=1, sticky="ew", padx=(0, 4), pady=4)
        ttk.Button(frame, text="찾아보기",
                   command=lambda: var.set(
                       filedialog.askopenfilename(title=title, filetypes=filetypes)
                       or var.get()
                   )).grid(row=row, column=2, padx=(0, 12), pady=4)

    video_var = tk.StringVar()
    srt_var = tk.StringVar()
    thumb_var = tk.StringVar()
    out_var = tk.StringVar()

    browse_row(0, "영상 파일", video_var, "영상 파일 선택",
               [("동영상", "*.mp4 *.mov *.mkv *.avi *.webm"), ("전체", "*.*")])
    browse_row(1, "자막 파일 (.srt)", srt_var, "자막 파일 선택",
               [("자막", "*.srt *.ass"), ("전체", "*.*")])
    browse_row(2, "썸네일 이미지", thumb_var, "썸네일 이미지 선택",
               [("이미지", "*.jpg *.jpeg *.png *.webp *.bmp"), ("전체", "*.*")])

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
            out_var.set(base + "_완성.mp4")
    video_var.trace_add("write", autofill)

    # 출력 파일
    ttk.Label(frame, text="출력 파일").grid(row=3, column=0, sticky="w", **pad)
    ttk.Entry(frame, textvariable=out_var, width=52).grid(
        row=3, column=1, sticky="ew", padx=(0, 4), pady=4)
    ttk.Button(frame, text="저장 위치",
               command=lambda: out_var.set(
                   filedialog.asksaveasfilename(
                       title="완성 영상 저장", defaultextension=".mp4",
                       filetypes=[("MP4 영상", "*.mp4")]) or out_var.get()
               )).grid(row=3, column=2, padx=(0, 12), pady=4)

    # 옵션
    opt = ttk.LabelFrame(frame, text="옵션", padding=8)
    opt.grid(row=4, column=0, columnspan=3, sticky="ew", padx=12, pady=6)
    opt.columnconfigure(1, weight=1)
    opt.columnconfigure(3, weight=1)

    intro_var = tk.BooleanVar(value=True)
    cover_var = tk.BooleanVar(value=True)
    burn_var = tk.BooleanVar(value=True)
    intro_sec_var = tk.StringVar(value="2.5")
    font_size_var = tk.StringVar(value="24")

    ttk.Checkbutton(opt, text="썸네일 인트로 붙이기",
                    variable=intro_var).grid(row=0, column=0, sticky="w", padx=8, pady=3)
    ttk.Label(opt, text="인트로 길이 (초)").grid(row=0, column=2, sticky="w", padx=(16, 4), pady=3)
    ttk.Entry(opt, textvariable=intro_sec_var, width=6).grid(row=0, column=3, sticky="w", pady=3)

    ttk.Checkbutton(opt, text="썸네일 표지(커버) 삽입",
                    variable=cover_var).grid(row=1, column=0, sticky="w", padx=8, pady=3)
    ttk.Label(opt, text="자막 크기").grid(row=1, column=2, sticky="w", padx=(16, 4), pady=3)
    ttk.Entry(opt, textvariable=font_size_var, width=6).grid(row=1, column=3, sticky="w", pady=3)

    ttk.Checkbutton(opt, text="자막 영상에 새겨넣기(하드섭)",
                    variable=burn_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=3)

    # 실행 버튼
    btn_label_var = tk.StringVar(value="완성 영상 만들기")
    run_btn = ttk.Button(frame, textvariable=btn_label_var, style="Accent.TButton")
    run_btn.grid(row=5, column=0, columnspan=3, padx=12, pady=8, sticky="ew")

    # 로그
    log = make_log(frame)
    log.grid(row=6, column=0, columnspan=3, sticky="nsew", padx=12, pady=(0, 12))

    def on_run():
        video = video_var.get().strip()
        srt = srt_var.get().strip()
        thumb = thumb_var.get().strip()
        out = out_var.get().strip()
        if not video:
            messagebox.showwarning("입력 오류", "영상 파일을 선택하세요.")
            return
        if burn_var.get() and not srt:
            messagebox.showwarning("입력 오류", "자막 파일을 선택하세요.\n(자막 새겨넣기를 끄면 자막 없이 진행됩니다.)")
            return
        if (intro_var.get() or cover_var.get()) and not thumb:
            messagebox.showwarning("입력 오류", "썸네일 이미지를 선택하세요.\n(인트로/표지 옵션을 모두 끄면 썸네일 없이 진행됩니다.)")
            return
        if not out:
            messagebox.showwarning("입력 오류", "출력 파일 경로를 지정하세요.")
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

        log.config(state="normal")
        log.delete("1.0", tk.END)
        log.config(state="disabled")
        run_btn.config(state="disabled")
        btn_label_var.set("처리 중...")

        def done(ok):
            run_btn.config(state="normal")
            btn_label_var.set("완성 영상 만들기")
            if ok:
                messagebox.showinfo("완료", f"완성 영상이 저장되었습니다.\n\n{out}")
            else:
                messagebox.showerror("오류", "처리 중 오류가 발생했습니다.\n로그를 확인하세요.")

        run_script(cmd, log, done)

    run_btn.config(command=on_run)
    return frame


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.title("YouTube 영상 요약기")
    root.geometry("720x640")
    root.minsize(640, 560)

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
    root.configure(bg="#2d2d2d")

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)

    build_summarizer_tab(nb)
    build_finalize_tab(nb)

    root.mainloop()


if __name__ == "__main__":
    main()

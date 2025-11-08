import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = BASE_DIR / "run.py"


class AnalyzerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Audio Rubric Runner")
        self.audio_path: Optional[str] = None
        self.process_thread: Optional[threading.Thread] = None
        self.running = False
        self.output_queue: queue.Queue = queue.Queue()

        self.selected_file_var = tk.StringVar(value="No file selected")

        intro = tk.Label(
            root,
            text="Select a .wav or .mp3 file, then click Run to send it to the evaluator.",
            wraplength=480,
            justify="left",
        )
        intro.pack(padx=12, pady=(12, 6), anchor="w")

        controls = tk.Frame(root)
        controls.pack(fill="x", padx=12)

        select_btn = tk.Button(controls, text="Select Audio", command=self.select_file)
        select_btn.pack(side="left")

        self.file_label = tk.Label(controls, textvariable=self.selected_file_var, anchor="w")
        self.file_label.pack(side="left", padx=8, fill="x", expand=True)

        self.run_button = tk.Button(root, text="Run", command=self.run_analysis, state="disabled", width=10)
        self.run_button.pack(padx=12, pady=(6, 12), anchor="e")

        self.log_text = scrolledtext.ScrolledText(root, height=20, width=80)
        self.log_text.configure(font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.root.after(100, self._poll_queue)

    def select_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio Files", "*.wav *.mp3"), ("All Files", "*.*")],
        )
        if path:
            self.audio_path = path
            self.selected_file_var.set(path)
            self.run_button.config(state="normal")

    def run_analysis(self) -> None:
        if not RUN_SCRIPT.exists():
            messagebox.showerror("Missing Script", f"Could not find run.py at {RUN_SCRIPT}")
            return
        if not self.audio_path:
            messagebox.showwarning("No Audio", "Please select an audio file first.")
            return
        if self.running:
            return
        if not os.path.exists(self.audio_path):
            messagebox.showerror("File Missing", "The selected audio file no longer exists.")
            return

        self.log_text.delete("1.0", tk.END)
        self.running = True
        self.run_button.config(state="disabled")

        self.process_thread = threading.Thread(target=self._run_process, daemon=True)
        self.process_thread.start()

    def _run_process(self) -> None:
        cmd = [sys.executable, "run.py", "--audio", self.audio_path]
        try:
            process = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            self.output_queue.put("Failed to start python interpreter.\n")
            self.output_queue.put(None)
            return

        assert process.stdout is not None
        for line in process.stdout:
            self.output_queue.put(line)
        return_code = process.wait()
        self.output_queue.put(f"\n[process] Finished with exit code {return_code}\n")
        self.output_queue.put(None)

    def _poll_queue(self) -> None:
        try:
            while True:
                message = self.output_queue.get_nowait()
                if message is None:
                    self.running = False
                    self.run_button.config(state="normal")
                    break
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_queue)


def main() -> None:
    root = tk.Tk()
    AnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

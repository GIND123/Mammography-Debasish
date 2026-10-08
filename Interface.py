"""DiceMed desktop tool: ACR breast-density estimation from a CC + MLO mammogram pair.

The heavy lifting (preprocessing, ensemble inference, calibration, guardrails,
Grad-CAM) lives in mammo/inference.py; this file is only the user interface.
"""
import json
import os
import sys
import threading
from datetime import datetime
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageOps

FILETYPES = [("Mammograms", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.dcm"), ("All files", "*.*")]
SEVERITY_COLOURS = {"block": "#f87171", "warn": "#fbbf24", "info": "#cbd5e1"}
CLASS_COLOURS = {"A": "#60a5fa", "B": "#34d399", "C": "#fbbf24", "D": "#f87171"}


def resource_path(rel):
    """Path to a bundled resource, both from source and inside a PyInstaller executable."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


def fit(img: Image.Image, w: int, h: int, bg=(30, 59, 139)) -> Image.Image:
    img = ImageOps.contain(img.convert("RGB"), (w, h))
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return canvas


class MammographyTool(ctk.CTk):
    BG_COLOR = "#1E3B8B"
    PANEL = "#16306f"

    def __init__(self):
        super().__init__()
        self.title("DiceMed - Mammographic Density (research prototype)")
        self.geometry("900x600")
        self.resizable(False, False)
        self.bg_image_path = resource_path("Background.png")
        self.colour_image_path = resource_path("bgcolour.png")
        self.predictors = {}  # mode -> mammo.inference.MammoPredictor, loaded in the background
        self.predictor_error: str | None = None
        self.accurate_mode = ctk.BooleanVar(value=os.environ.get("DICEMED_MODE") == "accurate")
        self.last_result = None
        threading.Thread(target=self._load_predictor, args=("fast",), daemon=True).start()  # warm up while user uploads
        self.show_home()

    # ------------------------------------------------------------------ model
    @property
    def predictor(self):
        return self.predictors.get("fast")

    def _load_predictor(self, mode):
        """Fast mode: 2 networks (validated pairing). Accurate mode: all 10 fold networks.
        Uses a CUDA GPU automatically when PyTorch can see one."""
        try:
            from mammo.inference import MammoPredictor

            bundle = os.environ.get("DICEMED_BUNDLE") or resource_path(os.path.join("models", "dicemed_density_v2.pt"))
            self.predictors[mode] = MammoPredictor(bundle, device=None, mode=mode)
        except Exception as e:  # surfaced to the user when they press Predict
            self.predictor_error = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ pages
    def _clear(self, background):
        for widget in self.winfo_children():
            widget.destroy()
        bg_img = Image.open(background).resize((900, 600))
        self._bg = ctk.CTkImage(light_image=bg_img, size=(900, 600))
        ctk.CTkLabel(self, image=self._bg, text="").place(x=0, y=0, relwidth=1, relheight=1)

    def show_home(self):
        self._clear(self.bg_image_path)
        ctk.CTkButton(self, text="Next", fg_color="#16a34a", text_color="white", height=50, width=160,
                      command=self.show_upload_page).place(relx=0.5, rely=0.92, anchor="center")

    def show_upload_page(self):
        self._clear(self.colour_image_path)
        ctk.CTkButton(self, text="Home", width=80, command=self.show_home).place(x=10, y=10)
        self.paths: dict[str, str | None] = {"CC": None, "MLO": None}
        self.thumbs = {}
        self.thumb_labels = {}
        for view, x in (("CC", 200), ("MLO", 540)):
            ctk.CTkButton(self, text=f"Upload {view}", width=160, height=40, fg_color="#2563eb", text_color="white",
                          command=lambda v=view: self.upload(v)).place(x=x, y=90, anchor="n")
            lbl = ctk.CTkLabel(self, text=f"No {view} image", width=250, height=330, fg_color=self.PANEL,
                               text_color="white", corner_radius=6)
            lbl.place(x=x, y=145, anchor="n")
            self.thumb_labels[view] = lbl
        self.status = ctk.CTkLabel(self, text="Upload the CC and MLO views of the same breast.",
                                   text_color="white", fg_color=self.BG_COLOR, font=("Arial", 13))
        self.status.place(relx=0.5, rely=0.84, anchor="center")
        self.predict_btn = ctk.CTkButton(self, text="Predict", width=200, height=50, fg_color="#16a34a",
                                         text_color="white", command=self.predict, state="disabled")
        self.predict_btn.place(relx=0.5, rely=0.93, anchor="center")
        ctk.CTkSwitch(self, text="High-accuracy mode (10 models, slower)", variable=self.accurate_mode,
                      text_color="white", bg_color=self.BG_COLOR).place(x=15, rely=0.93, anchor="w")

    def upload(self, view):
        path = filedialog.askopenfilename(title=f"Select {view} image", filetypes=FILETYPES)
        if not path:
            return
        self.paths[view] = path
        try:
            if path.lower().endswith((".dcm", ".dicom")):
                from mammo.preprocess import load_grayscale

                img = Image.fromarray((load_grayscale(path) * 255).astype("uint8"))
            else:
                img = Image.open(path)
            self.thumbs[view] = ctk.CTkImage(light_image=fit(img, 250, 330, bg=(22, 48, 111)), size=(250, 330))
            self.thumb_labels[view].configure(image=self.thumbs[view], text="")
        except Exception:
            self.thumb_labels[view].configure(image=None, text=f"{os.path.basename(path)}\n(preview unavailable)")
        ready = all(self.paths.values())
        self.predict_btn.configure(state="normal" if ready else "disabled")

    def predict(self):
        self.predict_btn.configure(state="disabled")
        self.status.configure(text="Analysing...")
        threading.Thread(target=self._predict_worker, args=(dict(self.paths),), daemon=True).start()

    def _predict_worker(self, paths):
        import time

        mode = "accurate" if self.accurate_mode.get() else "fast"
        if mode not in self.predictors and mode != "fast":
            self._load_predictor(mode)
        while mode not in self.predictors and self.predictor_error is None:
            time.sleep(0.1)
        if self.predictor_error:
            self.after(0, lambda: self._error(self.predictor_error))
            return
        try:
            result = self.predictors[mode].predict(paths["CC"], paths["MLO"], explain=True)
        except Exception as e:
            self.after(0, lambda: self._error(f"{type(e).__name__}: {e}"))
            return
        self.after(0, lambda: self.show_result_page(result, paths))

    def _error(self, msg):
        self.status.configure(text=f"Error: {msg}", text_color="#fca5a5")
        self.predict_btn.configure(state="normal")

    def show_result_page(self, r, paths):
        self.last_result, self.last_paths = r, paths
        self._clear(self.colour_image_path)
        ctk.CTkButton(self, text="Home", width=80, command=self.show_home).place(x=10, y=10)
        self.result_imgs = []
        for i, view in enumerate(("CC", "MLO")):
            if view in r.overlays:
                img = Image.fromarray(r.overlays[view])
            else:
                try:
                    img = Image.open(paths[view])
                except Exception:
                    img = Image.new("RGB", (250, 330), (30, 59, 139))
            ph = ctk.CTkImage(light_image=fit(img, 265, 375), size=(265, 375))
            self.result_imgs.append(ph)
            ctk.CTkLabel(self, image=ph, text="").place(x=15 + i * 278, y=55)
            cap = view + (" - Grad-CAM" if view in r.overlays else "")
            if r.status == "ok":
                pv = r.per_view.get(view, {})
                best = max(pv, key=pv.get) if pv else "?"
                cap += f"   ({view} alone: {best} {pv.get(best, 0):.0%})"
            ctk.CTkLabel(self, text=cap, text_color="white", fg_color=self.BG_COLOR, font=("Arial", 12)).place(
                x=15 + i * 278, y=435)
        panel = ctk.CTkFrame(self, width=325, height=470, fg_color=self.PANEL, corner_radius=10)
        panel.place(x=565, y=50)
        if r.status == "ok":
            ctk.CTkLabel(panel, text=f"ACR density  {r.density}", font=("Arial", 26, "bold"),
                         text_color=CLASS_COLOURS.get(r.density, "white")).place(x=15, y=10)
            ctk.CTkLabel(panel, text=r.description, font=("Arial", 12), text_color="white", wraplength=295,
                         justify="left").place(x=15, y=50)
            y = 85
            for c, p in r.probabilities.items():
                ctk.CTkLabel(panel, text=c, text_color="white", font=("Arial", 13, "bold")).place(x=15, y=y)
                bar = ctk.CTkProgressBar(panel, width=210, height=14, progress_color=CLASS_COLOURS[c])
                bar.set(p)
                bar.place(x=40, y=y + 7)
                ctk.CTkLabel(panel, text=f"{p:.0%}", text_color="white").place(x=260, y=y)
                y += 28
            ctk.CTkLabel(panel, text=f"Dense breast (C/D) probability: {r.dense_probability:.0%}\n"
                                     f"Confidence: {r.confidence:.0%}", text_color="white", justify="left",
                         font=("Arial", 12)).place(x=15, y=y + 2)
            y += 45
            if r.suspicion:
                s = r.suspicion
                ctk.CTkLabel(panel, text=f"Suspicion score (experimental): {s['score']:.2f}"
                                         f"{'  - review advised' if s['flagged'] else ''}",
                             text_color="#fbbf24" if s["flagged"] else "white", font=("Arial", 12)).place(x=15, y=y)
                y += 25
        else:
            ctk.CTkLabel(panel, text="No prediction", font=("Arial", 24, "bold"), text_color="#f87171").place(x=15, y=10)
            ctk.CTkLabel(panel, text="The input did not pass the safety checks:", text_color="white").place(x=15, y=50)
            y = 80
        box = ctk.CTkTextbox(panel, width=300, height=max(80, 455 - y), fg_color="#0f2557", text_color="white",
                             font=("Arial", 11), wrap="word")
        box.place(x=12, y=y)
        if not r.findings:
            box.insert("end", "All input checks passed.\n")
        for f in r.findings:
            tag = f["severity"]
            box.insert("end", f"[{tag.upper()}] {f['message']}\n\n", tag)
            box.tag_config(tag, foreground=SEVERITY_COLOURS[tag])
        box.configure(state="disabled")
        ctk.CTkLabel(self, text=r.disclaimer, text_color="#cbd5e1", fg_color=self.BG_COLOR, font=("Arial", 10),
                     wraplength=530, justify="left").place(x=15, y=468)
        ctk.CTkButton(self, text="New case", fg_color="#16a34a", width=160, height=42,
                      command=self.show_upload_page).place(relx=0.38, rely=0.93, anchor="center")
        ctk.CTkButton(self, text="Save report", fg_color="#2563eb", width=160, height=42,
                      command=self.save_report).place(relx=0.62, rely=0.93, anchor="center")
        ctk.CTkButton(self, text="Exit", width=80, fg_color="#dc2626", hover_color="#b91c1c",
                      command=self.destroy).place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

    def save_report(self):
        r = self.last_result
        if r is None:
            return
        stem = filedialog.asksaveasfilename(title="Save report", defaultextension=".png",
                                            initialfile=f"dicemed_report_{datetime.now():%Y%m%d_%H%M%S}.png",
                                            filetypes=[("PNG", "*.png")])
        if not stem:
            return
        # No patient identifiers are written: only file names chosen by the user, results and checks.
        payload = dict(created=datetime.now().isoformat(timespec="seconds"), model_version=r.model_version,
                       status=r.status, density=r.density, probabilities=r.probabilities,
                       dense_probability=r.dense_probability, confidence=r.confidence, per_view=r.per_view,
                       suspicion=r.suspicion, ood=r.ood, findings=r.findings, disclaimer=r.disclaimer,
                       inputs={v: os.path.basename(p) for v, p in self.last_paths.items()})
        with open(os.path.splitext(stem)[0] + ".json", "w") as f:
            json.dump(payload, f, indent=1)
        tiles = [fit(Image.fromarray(r.overlays[v]) if v in r.overlays else Image.new("RGB", (10, 10)), 400, 520)
                 for v in ("CC", "MLO")]
        sheet = Image.new("RGB", (820, 700), (255, 255, 255))
        for i, t in enumerate(tiles):
            sheet.paste(t, (10 + i * 405, 10))
        d = ImageDraw.Draw(sheet)
        lines = [f"Status: {r.status}   Model: {r.model_version}"]
        if r.status == "ok":
            lines.append(f"ACR density {r.density} ({r.description})  confidence {r.confidence:.0%}  "
                         f"P(dense)={r.dense_probability:.0%}")
            lines.append("  ".join(f"{k}: {v:.0%}" for k, v in r.probabilities.items()))
        lines += [f"[{f['severity']}] {f['message']}"[:130] for f in r.findings[:5]]
        lines.append(r.disclaimer[:130])
        for i, line in enumerate(lines):
            d.text((10, 540 + i * 18), line, fill=(0, 0, 0))
        sheet.save(stem)


if __name__ == "__main__":
    app = MammographyTool()
    app.mainloop()

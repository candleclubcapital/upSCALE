# pip install torch diffusers transformers accelerate safetensors Pillow PySide6

import os, sys
from pathlib import Path
from PIL import Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QPushButton, QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFileDialog, QProgressBar, QComboBox, QLabel
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor


# ============================================================
#                   Worker Thread
# ============================================================
class UpscaleWorker(QThread):
    log = Signal(str)
    progress = Signal(int)
    finished = Signal()

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def run(self):
        c = self.cfg
        try:
            refiner = None
            if c["use_refiner"]:
                self.log.emit(f"[INIT] Loading refiner: {c['refiner_model']} ({c['device']})")
                refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    c["refiner_model"],
                    torch_dtype=torch.float16 if c["device"] != "cpu" else torch.float32,
                    use_safetensors=True
                ).to(c["device"])
                refiner.set_progress_bar_config(disable=True)
            else:
                self.log.emit("[INFO] Refiner disabled - using standard upscaling only")

            files = [f for f in Path(c["input"]).iterdir()
                     if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]]
            total = len(files)
            if not total:
                self.log.emit("[ERROR] No images found.")
                self.finished.emit()
                return
            Path(c["output"]).mkdir(parents=True, exist_ok=True)

            for i, fpath in enumerate(files, 1):
                if self.stop_flag:
                    break
                self.log.emit(f"[PROCESS] ({i}/{total}) {fpath.name}")
                try:
                    img = Image.open(fpath).convert("RGB")
                    orig_w, orig_h = img.size

                    new_w = int(orig_w * c["scale_factor"])
                    new_h = int(orig_h * c["scale_factor"])

                    if c["max_dimension"] > 0:
                        if new_w > c["max_dimension"] or new_h > c["max_dimension"]:
                            ratio = min(c["max_dimension"] / new_w, c["max_dimension"] / new_h)
                            new_w = int(new_w * ratio)
                            new_h = int(new_h * ratio)

                    self.log.emit(f"  ↳ Upscaling {orig_w}x{orig_h} → {new_w}x{new_h}")

                    resample_map = {
                        "Lanczos": Image.Resampling.LANCZOS,
                        "Bicubic": Image.Resampling.BICUBIC,
                        "Bilinear": Image.Resampling.BILINEAR,
                        "Nearest": Image.Resampling.NEAREST
                    }
                    resample = resample_map.get(c["resample_method"], Image.Resampling.LANCZOS)
                    upscaled = img.resize((new_w, new_h), resample)

                    if c["use_refiner"]:
                        self.log.emit("  ↳ [REFINER] Enhancing details…")
                        prompt = c.get("prompt", "high quality, detailed, sharp")
                        neg_prompt = c.get("neg_prompt", "blurry, low quality, distorted")

                        refined = refiner(
                            prompt=prompt,
                            negative_prompt=neg_prompt,
                            image=upscaled,
                            strength=c["refiner_strength"],
                            guidance_scale=c["guidance_scale"],
                            num_inference_steps=c["refiner_steps"]
                        ).images[0]
                        output = refined
                    else:
                        output = upscaled

                    suffix = "_upscaled" if not c["use_refiner"] else "_upscaled_refined"
                    output_path = Path(c["output"]) / f"{fpath.stem}{suffix}.png"
                    output.save(output_path, quality=c["output_quality"])
                    self.log.emit(f"  ✓ Saved {output_path.name}")

                except Exception as e:
                    self.log.emit(f"[ERROR] {fpath.name}: {e}")
                self.progress.emit(int(i / total * 100))

            if refiner:
                del refiner
                torch.cuda.empty_cache()
            self.log.emit("[DONE] All images processed.")
        except Exception as e:
            self.log.emit(f"[FATAL] {e}")
        self.finished.emit()


# ============================================================
#                   GUI
# ============================================================
class UpscalerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("upSCALE")
        self.setMinimumSize(1200, 750)
        self.device = self.detect_device()
        self.worker = None
        self.build_ui()
        self.apply_style()

    def detect_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def build_ui(self):
        root = QHBoxLayout(self)
        left = QVBoxLayout()

        # --- Directories ---
        dirs = QGroupBox("Directories")
        f = QFormLayout()
        self.in_dir, self.out_dir = QLineEdit(), QLineEdit()
        binp, bout = QPushButton("Browse"), QPushButton("Browse")
        binp.clicked.connect(lambda: self.pick_folder(self.in_dir))
        bout.clicked.connect(lambda: self.pick_folder(self.out_dir))
        r1 = QHBoxLayout(); r1.addWidget(self.in_dir); r1.addWidget(binp)
        r2 = QHBoxLayout(); r2.addWidget(self.out_dir); r2.addWidget(bout)
        f.addRow("Input:", r1)
        f.addRow("Output:", r2)
        dirs.setLayout(f)
        left.addWidget(dirs)

        # --- Upscale Settings ---
        upscale = QGroupBox("Upscaling Settings")
        fu = QFormLayout()
        self.scale_factor = QDoubleSpinBox()
        self.scale_factor.setRange(1.0, 8.0)
        self.scale_factor.setSingleStep(0.5)
        self.scale_factor.setValue(2.0)

        self.max_dimension = QSpinBox()
        self.max_dimension.setRange(0, 8192)
        self.max_dimension.setValue(4096)
        self.max_dimension.setSpecialValueText("No Limit")

        self.resample_method = QComboBox()
        self.resample_method.addItems(["Lanczos", "Bicubic", "Bilinear", "Nearest"])

        self.output_quality = QSpinBox()
        self.output_quality.setRange(1, 100)
        self.output_quality.setValue(95)

        fu.addRow("Scale Factor:", self.scale_factor)
        fu.addRow("Max Dimension:", self.max_dimension)
        fu.addRow("Resample Method:", self.resample_method)
        fu.addRow("Output Quality:", self.output_quality)
        upscale.setLayout(fu)
        left.addWidget(upscale)

        # --- Refiner Settings ---
        refiner = QGroupBox("Refiner (Optional)")
        fr = QFormLayout()
        self.use_refiner = QCheckBox("Enable AI Refinement")
        self.use_refiner.setChecked(False)
        self.use_refiner.toggled.connect(self.toggle_refiner_settings)
        self.refiner_model = QLineEdit("stabilityai/stable-diffusion-xl-refiner-1.0")

        self.refiner_strength = QDoubleSpinBox()
        self.refiner_strength.setRange(0.0, 1.0)
        self.refiner_strength.setSingleStep(0.05)
        self.refiner_strength.setValue(0.3)

        self.guidance_scale = QDoubleSpinBox()
        self.guidance_scale.setRange(1.0, 30.0)
        self.guidance_scale.setValue(7.5)

        self.refiner_steps = QSpinBox()
        self.refiner_steps.setRange(1, 100)
        self.refiner_steps.setValue(15)

        fr.addRow("", self.use_refiner)
        fr.addRow("Model:", self.refiner_model)
        fr.addRow("Strength:", self.refiner_strength)
        fr.addRow("Guidance Scale:", self.guidance_scale)
        fr.addRow("Steps:", self.refiner_steps)
        refiner.setLayout(fr)
        left.addWidget(refiner)

        # --- Controls ---
        ctl = QHBoxLayout()
        self.start = QPushButton("START UPSCALING")
        self.stop = QPushButton("STOP")
        self.stop.setEnabled(False)
        self.start.clicked.connect(self.start_batch)
        self.stop.clicked.connect(self.stop_batch)
        ctl.addWidget(self.start)
        ctl.addWidget(self.stop)
        left.addLayout(ctl)
        root.addLayout(left, 1)

        # ==== RIGHT COLUMN ====
        right = QVBoxLayout()

        # --- Refiner Prompts ---
        prompts = QGroupBox("Refiner Prompts (Optional)")
        pf = QFormLayout()
        self.prompt = QTextEdit("high quality, detailed, sharp, clear")
        self.prompt.setMaximumHeight(70)
        self.neg_prompt = QTextEdit("blurry, low quality, distorted, artifacts, noise")
        self.neg_prompt.setMaximumHeight(60)
        pf.addRow("Prompt:", self.prompt)
        pf.addRow("Negative:", self.neg_prompt)
        prompts.setLayout(pf)
        right.addWidget(prompts)

        # --- Info ---
        info = QGroupBox("Info")
        info_layout = QVBoxLayout()
        info_label = QLabel(
            f"Device: {self.device.upper()}\n"
            "Supported formats: PNG, JPG, JPEG, WEBP, BMP\n"
            "Refiner uses SDXL for AI-enhanced detail recovery"
        )
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        info.setLayout(info_layout)
        right.addWidget(info)

        # --- Progress + Log ---
        self.progress = QProgressBar()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        right.addWidget(self.progress)
        right.addWidget(self.log, 5)

        root.addLayout(right, 1)
        self.setLayout(root)

        # ✅ Fix: now safe to disable refiner settings AFTER prompt creation
        self.toggle_refiner_settings(False)

    def toggle_refiner_settings(self, enabled):
        self.refiner_model.setEnabled(enabled)
        self.refiner_strength.setEnabled(enabled)
        self.guidance_scale.setEnabled(enabled)
        self.refiner_steps.setEnabled(enabled)
        if hasattr(self, "prompt"):
            self.prompt.setEnabled(enabled)
            self.neg_prompt.setEnabled(enabled)

    def apply_style(self):
        self.setStyleSheet("""
            QWidget { background-color:#0a0f0a; color:#00ff99; font-family:'Courier New'; font-size:12px; }
            QGroupBox { border:1px solid #00ff99; border-radius:6px; margin-top:6px; padding:6px; font-weight:bold; }
            QPushButton { background-color:#001a0f; border:1px solid #00ff99; border-radius:6px; padding:6px 10px; }
            QPushButton:hover { background-color:#00331a; }
            QPushButton:disabled { background-color:#0a0f0a; color:#004d26; border-color:#004d26; }
            QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color:#001a0f; border:1px solid #004d26; border-radius:4px; color:#00ff99; padding:4px;
            }
            QTextEdit:disabled, QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
                color:#004d26; border-color:#003319;
            }
            QComboBox::drop-down { border:none; }
            QComboBox::down-arrow { width:0; height:0; }
            QProgressBar { border:1px solid #00ff99; border-radius:4px; height:16px; text-align:center; color:black; }
            QProgressBar::chunk { background-color:#00ff99; margin:0.5px; }
            QCheckBox::indicator { width:14px; height:14px; }
            QCheckBox::indicator:checked { background-color:#00ff99; }
            QCheckBox::indicator:unchecked { border:1px solid #00ff99; }
            QLabel { color:#00ff99; }
        """)

    def logmsg(self, msg):
        self.log.append(msg)
        try:
            self.log.moveCursor(QTextCursor.End)
        except Exception:
            cur = self.log.textCursor()
            cur.movePosition(QTextCursor.End)
            self.log.setTextCursor(cur)
        QApplication.processEvents()

    def pick_folder(self, target):
        p = QFileDialog.getExistingDirectory(self, "Select Folder")
        if p:
            target.setText(p)

    def start_batch(self):
        if not self.in_dir.text() or not self.out_dir.text():
            self.logmsg("[ERROR] Select input/output folders.")
            return

        cfg = dict(
            input=self.in_dir.text().strip(),
            output=self.out_dir.text().strip(),
            scale_factor=self.scale_factor.value(),
            max_dimension=self.max_dimension.value(),
            resample_method=self.resample_method.currentText(),
            output_quality=self.output_quality.value(),
            use_refiner=self.use_refiner.isChecked(),
            refiner_model=self.refiner_model.text().strip(),
            refiner_strength=self.refiner_strength.value(),
            guidance_scale=self.guidance_scale.value(),
            refiner_steps=self.refiner_steps.value(),
            prompt=self.prompt.toPlainText().strip(),
            neg_prompt=self.neg_prompt.toPlainText().strip(),
            device=self.device
        )

        self.worker = UpscaleWorker(cfg)
        self.worker.log.connect(self.logmsg)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.finished_batch)
        self.start.setEnabled(False)
        self.stop.setEnabled(True)
        self.progress.setValue(0)
        self.logmsg(f"[START] Device: {self.device.upper()} | Scale: {cfg['scale_factor']}x | Refiner: {'ON' if cfg['use_refiner'] else 'OFF'}")
        self.worker.start()

    def stop_batch(self):
        if self.worker:
            self.worker.stop()
            self.logmsg("[STOP] Stop signal sent.")

    def finished_batch(self):
        self.logmsg("[SYSTEM] Finished or stopped.")
        self.start.setEnabled(True)
        self.stop.setEnabled(False)


# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = UpscalerApp()
    w.show()
    sys.exit(app.exec())

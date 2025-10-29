# upSCALE

**upSCALE** is a Python-based image upscaling application designed for both traditional and AI-enhanced workflows. It uses **Pillow** for fast interpolation and optionally integrates **Stable Diffusion XL Refiner** for next-generation detail recovery. The interface, built with **PySide6**, provides real-time logging, progress tracking, and complete parameter control in a futuristic cyberpunk theme.
<img width="1187" height="732" alt="Screenshot 2025-10-29 at 2 23 46 PM" src="https://github.com/user-attachments/assets/9955389f-d03a-4197-9410-9a4aa1355556" />

---

### ✨ Features

* Batch upscale entire folders of images
* Adjustable **scale factor**, **max dimensions**, and **resampling method**
* Optional **AI refinement** using SDXL Refiner for enhanced detail
* Control over **refiner strength**, **guidance scale**, and **steps**
* Real-time **logging**, **progress tracking**, and safe stoppability
* Neon-styled GUI inspired by Solana cyber aesthetics

---

### 🧩 Requirements

```bash
pip install torch diffusers transformers accelerate safetensors Pillow PySide6
```

---

### 🚀 Usage

1. Launch the app:

   ```bash
   python upscale.py
   ```
2. Select input and output directories.
3. Configure upscaling parameters (scale, quality, etc.).
4. Optionally enable **AI Refinement** and set its parameters.
5. Click **START UPSCALING** to begin.

Output files will be saved with `_upscaled.png` or `_upscaled_refined.png` suffixes.

---

### ⚙️ Options

| Setting             | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| **Scale Factor**    | Multiplies image dimensions (e.g., 2×, 4×)                 |
| **Max Dimension**   | Caps image width/height to prevent GPU overload            |
| **Resample Method** | Choose from Lanczos, Bicubic, Bilinear, or Nearest         |
| **Output Quality**  | Controls PNG/JPG compression quality                       |
| **AI Refinement**   | Enhances detail using SDXL Refiner                         |
| **Prompts**         | Optional positive and negative conditioning for refinement |

---

### 🖥️ Interface Overview

* **Left Panel:** Directories, scaling settings, and AI options
* **Right Panel:** Refiner prompts, device info, progress bar, and live log

---

### 🧠 Notes

* Supports PNG, JPG, JPEG, WEBP, BMP formats
* Auto-detects best device (`cuda`, `mps`, or `cpu`)
* Works standalone or as a refinement companion for SDXL workflows

---


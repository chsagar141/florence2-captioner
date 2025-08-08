
# 🖼️ Florence-2 Batch Image Captioning Tool

This tool allows you to caption a batch of images using Microsoft's **Florence-2-large** vision-language model, all through a clean and simple **Gradio interface**.

---

## ✅ Features

- Supports **all image formats** (JPG, PNG, WEBP, HEIC, etc.)
- Uses **Florence-2-large** for high-quality captions
- Custom **prompt** support (e.g., `<CUTE_CAT>`, `<FASHION>`, `<MORE_DETAILED_CAPTION>`)
- Optional **trigger word** (e.g., `This_person01`) prepended to each caption file
- Outputs saved as `.png` and `.txt` files
- Downloads results as a **ZIP archive**
- GPU acceleration if available

---

## 🚀 How to Use

### 1. 🛠️ Install Requirements
Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

> Requires Python 3.9+ and PyTorch 2.1+ - Recommended Python 3.12.0

---

### 2. ▶️ Run the Script
Simply run the script:

```bash
python florence_captioner.py
```

Once it starts, it will:
- Load the Florence-2 model
- Launch a Gradio web interface (usually at `http://127.0.0.1:7860`)

---

### 3. 📤 Upload & Caption Images

#### Interface Workflow:
1. Upload one or more images in **any format**
2. Enter a **caption prompt** (or use default: `<MORE_DETAILED_CAPTION>`)
3. Optionally add a **trigger word** to prefix each caption file
4. Click **🖼️ Generate Captions**
5. Review results in the gallery

---

### 4. 💾 Save & 📦 Download Results

- Click **💾 Save Results** to store image-caption pairs locally
  - Images saved as `img_0001.png`, etc.
  - Captions saved as `img_0001.txt`, etc.
  - Saved in a folder called `Final/`
- Click **📦 Download ZIP** to get a zipped version (`Final.zip`)

---

## 📁 Output Example

```
Final/
├── img_0001.png
├── img_0001.txt  ← Contains: This_person01 A young woman smiling...
├── img_0002.png
├── img_0002.txt
```

---

## 🧠 Tips

- Use trigger words for LoRA/SDXL training (`This_person01`, `subject_001`, etc.)
- Prompt suggestions:
  - `<MORE_DETAILED_CAPTION>`
  - `<CINEMATIC>`
  - `<PRODUCT_DESCRIPTION>`
  - `<FASHION_LOOK>`
- You can use it offline **after the first model download**

---

## ❓ Troubleshooting

- **Model not loading?**
  - Make sure you have a stable internet connection for first-time model download.
- **CUDA out of memory?**
  - Use smaller batches, or run on CPU (slow).
- **HEIC or RAW formats failing?**
  - Ensure Pillow and libheif are installed correctly.

---

## 🧾 License

This tool uses Microsoft's Florence-2 model under its respective license. See the [Florence-2 on Hugging Face](https://huggingface.co/microsoft/Florence-2-large) for details.

---

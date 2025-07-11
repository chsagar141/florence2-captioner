import os
import shutil
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
import torch

# ========== CONFIG ==========
MAX_SIZE = 896
input_dir = "input"
preprocess_dir = "preprocess"
output_dir = "output"
supported_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

# ========== DEVICE CHECK ==========
if not torch.cuda.is_available():
    print("❌ GPU not available. This script requires CUDA-enabled GPU.")
    exit(1)

device = "cuda"
dtype = torch.float16

# ========== SETUP ==========
os.makedirs(input_dir, exist_ok=True)
os.makedirs(preprocess_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# ========== STEP 1: PREPROCESS ==========


def resize_image_keep_aspect(image, max_size):
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    ratio = min(max_size / w, max_size / h)
    return image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)


def preprocess_images():
    files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f.lower())[1] in supported_exts
    ])
    if not files:
        print("⚠️ No supported images found in 'input/' folder.")
        exit(1)

    for i, filename in enumerate(tqdm(files, desc="🔄 Preprocessing images")):
        input_path = os.path.join(input_dir, filename)
        try:
            image = Image.open(input_path).convert("RGB")
        except UnidentifiedImageError:
            print(f"❌ Skipping unreadable file: {filename}")
            continue

        resized = resize_image_keep_aspect(image, MAX_SIZE)
        output_path = os.path.join(preprocess_dir, f"{i+1}.png")
        resized.save(output_path, format="PNG")


# ========== STEP 2: LOAD MODEL ==========
print("📦 Loading Florence-2-large model...")
model_id = "microsoft/Florence-2-large"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    trust_remote_code=True
).to(device)
print("✅ Model loaded.\n")

# ========== STEP 3: CAPTION IMAGES ==========


def generate_caption(image):
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(images=image, text=prompt,
                       return_tensors="pt").to(device, dtype)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    decoded = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        decoded, task=prompt, image_size=image.size
    )
    return result.get("<MORE_DETAILED_CAPTION>", "").strip()


def caption_images():
    files = sorted([
        f for f in os.listdir(preprocess_dir)
        if f.lower().endswith(".png")
    ])

    for filename in tqdm(files, desc="🖼️ Generating captions"):
        image_path = os.path.join(preprocess_dir, filename)
        image = Image.open(image_path).convert("RGB")
        caption = generate_caption(image)

        out_image_path = os.path.join(output_dir, filename)
        out_text_path = os.path.join(
            output_dir, filename.replace(".png", ".txt"))

        image.save(out_image_path)
        with open(out_text_path, "w", encoding="utf-8") as f:
            f.write(caption)

    print("\n✅ Captioning complete. Results saved in 'output/'.")


# ========== MAIN ==========
if __name__ == "__main__":
    preprocess_images()
    caption_images()

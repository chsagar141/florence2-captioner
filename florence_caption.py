import os
import shutil
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import gradio as gr
from tqdm import tqdm

# ========== CONFIG ========== #
MAX_SIZE = 896
OUTPUT_FOLDER = "Final"

# ========== DEVICE CHECK ========== #
if not torch.cuda.is_available():
    print("⚠️ WARNING: GPU not available. Running on CPU, which will be very slow.")
    device = "cpu"
    dtype = torch.float32
else:
    device = "cuda"
    dtype = torch.float16
    print("✅ GPU detected. Using CUDA for processing.")

# ========== LOAD MODEL ========== #


def load_model():
    print("📦 Loading Florence-2-large model...")
    model_id = "microsoft/Florence-2-large"
    try:
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        exit(1)
    print("✅ Model loaded.\n")
    return processor, model


processor, model = load_model()

# ========== CORE FUNCTIONS ========== #


def resize_image_keep_aspect(image, max_size):
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    ratio = min(max_size / w, max_size / h)
    return image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)


def generate_captions_for_gallery(image_list, user_prompt, progress=gr.Progress(track_tqdm=True)):
    if not image_list:
        return [], []

    prompt = user_prompt.strip() or "<MORE_DETAILED_CAPTION>"
    results = []
    print(f"🔄 Processing {len(image_list)} images...")

    for image_data in progress.tqdm(image_list, desc="🖼️ Generating Captions"):
        try:
            image_pil = image_data[0] if isinstance(
                image_data, tuple) else image_data
            image_rgb = image_pil.convert("RGB")
            image_resized = resize_image_keep_aspect(image_rgb, MAX_SIZE)

            inputs = processor(images=image_resized,
                               text=prompt, return_tensors="pt")
            inputs = {k: v.to(device, dtype=dtype if v.dtype ==
                              torch.float else None) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False
                )

            decoded = processor.batch_decode(
                outputs, skip_special_tokens=False)[0]
            result = processor.post_process_generation(
                decoded, task=prompt, image_size=image_resized.size)
            caption = result.get(prompt, "Error: Could not generate caption.").replace(
                "<|endoftext|>", "").strip()

            results.append((image_pil, caption))

        except Exception as e:
            print(f"❌ Error during captioning: {e}")
            results.append((image_pil, "❌ Failed to generate caption."))

    print(f"\n✅ Completed captioning {len(results)} images.")
    return results, results


def save_results(results_to_save, trigger_word=""):
    if not results_to_save:
        return "⚠️ No results to save."

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    trigger_word = trigger_word.strip()
    num_saved = 0

    for i, (image, caption) in enumerate(results_to_save):
        try:
            img_path = os.path.join(OUTPUT_FOLDER, f"img_{i+1:04d}.png")
            txt_path = os.path.join(OUTPUT_FOLDER, f"img_{i+1:04d}.txt")

            image.save(img_path, "PNG")

            final_caption = f"{trigger_word} {caption}".strip(
            ) if trigger_word else caption
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(final_caption)

            num_saved += 1
        except Exception as e:
            print(f"❌ Error saving image {i+1}: {e}")

    return f"✅ Saved {num_saved} image-caption pairs to '{OUTPUT_FOLDER}' folder."


def zip_and_return_path():
    zip_path = OUTPUT_FOLDER + ".zip"
    try:
        shutil.make_archive(OUTPUT_FOLDER, 'zip', OUTPUT_FOLDER)
        return zip_path
    except Exception as e:
        print(f"❌ Failed to zip results: {e}")
        return None


# ========== GRADIO UI ========== #
if __name__ == "__main__":
    with gr.Blocks(theme=gr.themes.Soft()) as gui:
        gr.Markdown("# Florence-2 Batch Image Captioning")
        gr.Markdown(
            "Upload images → Set prompt/trigger → Generate → Save → Download")

        results_state = gr.State([])

        prompt_input = gr.Textbox(
            label="Caption Prompt",
            value="<MORE_DETAILED_CAPTION>",
            placeholder="e.g., <MORE_DETAILED_CAPTION>, <FASHION>, <CUTE_CAT>",
            interactive=True
        )

        trigger_input = gr.Textbox(
            label="Trigger Word (Optional)",
            placeholder="e.g. This_person01",
            interactive=True
        )

        with gr.Row():
            input_gallery = gr.Gallery(
                label="Upload Images (all formats supported)",
                type="pil",
                elem_id="gallery_input"
            )
            output_gallery = gr.Gallery(
                label="Captioned Results",
                show_label=True,
                elem_id="gallery_output",
                allow_preview=True
            )

        with gr.Row():
            generate_btn = gr.Button(
                "🖼️ Generate Captions", variant="primary", scale=2)
            save_btn = gr.Button("💾 Save Results", variant="secondary")
            zip_btn = gr.Button("📦 Download ZIP", variant="secondary")

        status_text = gr.Textbox(label="Status", interactive=False)
        zip_file = gr.File(label="Download ZIP", interactive=False)

        # Button Logic
        generate_btn.click(
            fn=generate_captions_for_gallery,
            inputs=[input_gallery, prompt_input],
            outputs=[output_gallery, results_state]
        )

        save_btn.click(
            fn=save_results,
            inputs=[results_state, trigger_input],
            outputs=status_text
        )

        zip_btn.click(
            fn=zip_and_return_path,
            outputs=zip_file
        )

    print("🚀 Launching Florence-2 Gradio App...")
    gui.launch(share=True)

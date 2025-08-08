import os
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


# ========== STEP 1: LOAD MODEL (Done only once at startup) ========== #
def load_model():
    """Loads the Florence-2 model and processor."""
    print("📦 Loading Florence-2-large model... (This may take a moment on first run)")
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
        print("Please ensure you have an internet connection to download the model.")
        exit(1)

    print("✅ Model loaded successfully.\n")
    return processor, model


# Load the model and processor globally to avoid reloading
processor, model = load_model()


# ========== STEP 2: DEFINE THE CORE FUNCTIONS ========== #
def resize_image_keep_aspect(image, max_size):
    """Resizes a PIL image while maintaining aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    ratio = min(max_size / w, max_size / h)
    return image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)


def generate_captions_for_gallery(image_list, progress=gr.Progress(track_tqdm=True)):
    """
    Takes a LIST of image data, generates captions, and returns the results
    for BOTH the output gallery and the hidden state variable.
    """
    if not image_list:
        return [], []  # Return empty for both outputs

    results = []
    print(f"🔄 Received {len(image_list)} images. Starting batch captioning...")

    for image_data in progress.tqdm(image_list, desc="🖼️ Generating Captions"):
        image_pil = image_data[0]  # Unpack the tuple to get the PIL image

        # --- Core Captioning Logic ---
        prompt = "<MORE_DETAILED_CAPTION>"
        image_rgb = image_pil.convert("RGB")
        image_resized = resize_image_keep_aspect(image_rgb, MAX_SIZE)

        inputs = processor(images=image_resized,
                           text=prompt, return_tensors="pt")
        inputs = {k: v.to(device, dtype=dtype if v.dtype ==
                          torch.float else None) for k, v in inputs.items()}

        outputs = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, num_beams=3, do_sample=False
        )

        decoded = processor.batch_decode(outputs, skip_special_tokens=False)[0]
        result = processor.post_process_generation(
            decoded, task=prompt, image_size=image_resized.size)
        caption = result.get("<MORE_DETAILED_CAPTION>",
                             "Error: Could not generate caption.").strip()

        results.append((image_pil, caption))

    print(f"\n✅ Batch captioning complete for {len(results)} images.")
    # Return results for the gallery and also for the state
    return results, results


def save_results(results_to_save):
    """
    Saves the images and captions from the state variable to the 'Final' folder.
    """
    if not results_to_save:
        return "⚠️ No results to save. Please generate captions first."

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    num_saved = 0
    for i, (image, caption) in enumerate(results_to_save):
        try:
            # Define file paths with sequential naming
            img_path = os.path.join(OUTPUT_FOLDER, f"img_{i+1}.png")
            txt_path = os.path.join(OUTPUT_FOLDER, f"img_{i+1}.txt")

            image.save(img_path, "PNG")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)
            num_saved += 1
        except Exception as e:
            print(f"Error saving file for image {i+1}: {e}")
            return f"❌ Error saving file for image {i+1}. Check console for details."

    return f"✅ Successfully saved {num_saved} image-caption pairs to the '{OUTPUT_FOLDER}' folder."


# ========== STEP 3: CREATE AND LAUNCH THE GRADIO GUI ========== #
if __name__ == "__main__":

    with gr.Blocks(theme=gr.themes.Soft()) as gui:
        gr.Markdown("# Batch Image Captioning with Florence-2")
        gr.Markdown(
            "Drop one or more images into the input box, click 'Generate', and then 'Save Results' to store them locally."
        )

        # Hidden state component to store the results between button clicks
        results_state = gr.State([])

        with gr.Row():
            input_gallery = gr.Gallery(
                label="Upload Images", type="pil", elem_id="gallery_input"
            )
            output_gallery = gr.Gallery(
                label="Captioned Images", show_label=True, elem_id="gallery_output", allow_preview=True
            )

        with gr.Row():
            generate_btn = gr.Button(
                "🖼️ Generate Captions", variant="primary", scale=2)
            save_btn = gr.Button(
                "💾 Save Results", variant="secondary", scale=1)

        status_text = gr.Textbox(label="Status", interactive=False)

        # Define button actions
        generate_btn.click(
            fn=generate_captions_for_gallery,
            inputs=input_gallery,
            # The output goes to two places: the visible gallery and the hidden state
            outputs=[output_gallery, results_state]
        )

        save_btn.click(
            fn=save_results,
            # The input for the save function is the data we stored in the hidden state
            inputs=results_state,
            # The output is the confirmation message in the status box
            outputs=status_text
        )

    # Launch the GUI
    print("🚀 Launching Gradio GUI...")
    print("Open the following URL in your browser to use the app.")
    gui.launch(share=True)

# Florence-2 Batch Image Captioning

A GPU-accelerated image captioning pipeline using Microsoft's [Florence-2-Large](https://huggingface.co/microsoft/Florence-2-large) vision-language model.

## Features
- Supports all common image formats (.jpg, .png, .webp, etc.)
- Automatically resizes to max 896×896 while preserving aspect ratio
- Converts all inputs to `.png`
- Uses <MORE_DETAILED_CAPTION> prompt for highest-quality captions
- Outputs matched `.png` and `.txt` files (e.g., `1.png`, `1.txt`)
- Uses only GPU — will exit if CUDA is not available

## Usage
1. Place raw images in the `input/` folder
2. Run the script:
   ```bash
   python florence_caption_pipeline.py

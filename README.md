# model-merge-engine
# Model & LoRA Merge Tool by UltraMuse (Optimized for Z-Image)

A premium, local Gradio-based web interface designed for easily merging two `.safetensors` model checkpoints or baking a LoRA directly into a stable diffusion / transformer checklist. 

This tool abstracts away the complex parameter math into a sleek frontend designed specifically for merging heavy neural networks with precise precision targeting (FP32, FP16, BF16).

---

## Features
- **Checkpoint Weighted Sum Blending**: Flexibly blend two base diffusion checkpoints according to a precise ratio (`α`).
- **LoRA Baking**: Permanently apply Low-Rank Adaptation (LoRA) weights into a base model to skip loading them at inference time.
- **Precision Override**: Optionally manually cast output tensors to FP32, FP16, or BF16 to save space or lock in formats required by target systems (such as single-stream DiTs).
- **Native File Picking**: Launch native Windows file-picker dialogue boxes straight from the browser to find massive local models easily.
- **Progress Tracking**: Realtime UI progress bar hooked directly to the tensor blending execution loop.

## Installation
Ensure you have the following requirements installed in your Python environment:
```bash
pip install torch gradio safetensors
```

## Running the UI
Navigate to the tool directory and execute the Python file via command line:
```bash
python app.py
```
Then open your exact local URL at http://127.0.0.1:7860 to begin merging.

---
*Developed by UltraMuse*

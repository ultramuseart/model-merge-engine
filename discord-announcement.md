# 🚀 Introducing: UltraMuse Model & LoRA Merge Engine

Hey everyone! 👋 We are completely thrilled to announce the launch of a brand-new internal tool we've been working on—the **UltraMuse Model Merge Engine**!

If you've ever struggled with trying to bake LoRAs directly into your base checkpoints, or tried carefully balancing two massive 12GB diffusion checkpoints together without crashing your entire system or losing architecture metadata, this tool is built specifically for you.

*Note: This tool was built and strictly tested using **Z-Image**, natively handling its heavy dimensions and S3-DiT architecture needs seamlessly. While we haven't tested it extensively on other architectures (like SDXL, SD 1.5, or Flux), the unified tensor math approach under the hood means it should theoretically still work!*

We stripped away all the bloated ComfyUI nodes, massive CLI scripts, and broken Python modules, and consolidated it into a sleek, premium, high-contrast local web interface. 

### ✨ Key Features
- **Checkpoint Blending**: Safely execute a mathematically perfect *Weighted Sum* blend of two base `.safetensors` checkpoints on the fly.
- **LoRA Baking Engine**: Permanently bake in your favorite LoRA weights directly into a base checkpoint layout, so you no longer have to load them separately.
- **Precision Overrides**: Force your output tensor weights directly into `FP32`, `FP16`, or `BF16` (ideal for native single-stream DiTs).
- **Bulletproof Architecture**: Built purely on PyTorch and Safetensors to ensure precision stability. It handles the heavy lifting, you just drag dropping.

---

## 🛠️ Getting Started in 60 Seconds

It runs entirely locally via **Gradio**. Here is how to boot it up:

**1. Clone the repository**
```bash
git clone https://github.com/ultramuseart/model-merge-engine.git
cd model-merge-engine
```

**2. Install the core requirements**
```bash
pip install torch gradio safetensors
```

**3. Launch the UltraMuse Engine**
```bash
python app.py
```

Once running, simply click the local IP address (`http://127.0.0.1:7860`) in your terminal to open the UI in your browser. From there, use the native file-pickers to grab your `.safetensors` files, adjust the Alpha Slider parameter, and hit **Execute Merge**!

We'd love for you to start merging your favorite styles and models and share what crazy fusions you come up with. Let us know if you run into any bugs or have feature requests! 🌌

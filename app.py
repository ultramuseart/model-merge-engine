import os
import torch
import gradio as gr
from safetensors.torch import load_file, save_file
import tkinter as tk
from tkinter import filedialog

def pick_file():
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Model File",
        filetypes=[("Safetensors", "*.safetensors"), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path

def pick_out_dir():
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Output Folder")
    root.destroy()
    if folder_path:
        return os.path.join(folder_path, "merged.safetensors")
    return "output/merged.safetensors"

def merge_models(model_a_path, model_b_path, merge_type, alpha, output_path, output_dtype, progress=gr.Progress()):
    if not os.path.exists(model_a_path):
        return f"Error: Cannot find Model A at {model_a_path}"
    if not os.path.exists(model_b_path):
        return f"Error: Cannot find Model B / LoRA at {model_b_path}"
    
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        progress(0.05, desc="Loading Checkpoint A...")
        print(f"Loading Checkpoint A from {model_a_path}...")
        sd_a = load_file(model_a_path)
        
        progress(0.15, desc="Loading Checkpoint B / LoRA...")
        print(f"Loading Checkpoint B / LoRA from {model_b_path}...")
        sd_b = load_file(model_b_path)
        
        merged_sd = {}
        
        if merge_type == "Checkpoint + Checkpoint (Weighted Sum)":
            progress(0.25, desc="Merging Checkpoints...")
            print(f"Merging Checkpoints with ratio: {alpha}...")
            # W = A * (1 - alpha) + B * alpha
            total_keys = len(sd_a.keys())
            for i, key in enumerate(sd_a.keys()):
                if i % max(1, total_keys // 100) == 0:
                    progress(0.25 + (0.50 * (i / total_keys)), desc=f"Blending Tensors... ({i}/{total_keys})")
                if key in sd_b:
                    weight_a = sd_a[key]
                    weight_b = sd_b[key]
                    
                    if weight_a.shape == weight_b.shape:
                        merged_sd[key] = (1.0 - alpha) * weight_a + alpha * weight_b
                    else:
                        print(f"Skipping {key} due to shape mismatch: {weight_a.shape} vs {weight_b.shape}")
                        merged_sd[key] = weight_a
                else:
                    merged_sd[key] = sd_a[key]
                    
        elif merge_type == "Checkpoint + LoRA":
            progress(0.25, desc="Parsing LoRA Modules...")
            print(f"Merging LoRA into Checkpoint with weight: {alpha}...")
            merged_sd = {k: v.clone() for k, v in sd_a.items()}
            
            # Map LoRA parts
            lora_modules = {}
            for lora_key in sd_b.keys():
                if '.lora_A.' in lora_key or '.lora_down.' in lora_key:
                    base_key = lora_key.replace('.lora_A.', '.').replace('.lora_down.', '.')
                    if base_key not in lora_modules: lora_modules[base_key] = {}
                    lora_modules[base_key]['down'] = sd_b[lora_key]
                elif '.lora_B.' in lora_key or '.lora_up.' in lora_key:
                    base_key = lora_key.replace('.lora_B.', '.').replace('.lora_up.', '.')
                    if base_key not in lora_modules: lora_modules[base_key] = {}
                    lora_modules[base_key]['up'] = sd_b[lora_key]
                elif '.alpha' in lora_key:
                    base_key = lora_key.replace('.alpha', '.weight')
                    if base_key not in lora_modules: lora_modules[base_key] = {}
                    lora_modules[base_key]['alpha'] = sd_b[lora_key]

            applied_count = 0
            total_mods = len(lora_modules.items())
            for i, (base_key, lora_parts) in enumerate(lora_modules.items()):
                if i % max(1, total_mods // 100) == 0:
                    progress(0.25 + (0.50 * (i / max(1, total_mods))), desc=f"Baking LoRA... ({i}/{total_mods})")
                    
                target_key = base_key
                # Attempt to handle diffusers/z_image architecture naming variations
                if target_key not in merged_sd:
                    alt_key = target_key.replace("diffusion_model.", "transformer.")
                    if alt_key in merged_sd:
                        target_key = alt_key
                
                if target_key in merged_sd and 'up' in lora_parts and 'down' in lora_parts:
                    up_weight = lora_parts['up'].to(torch.float32)
                    down_weight = lora_parts['down'].to(torch.float32)
                    
                    if len(up_weight.shape) == 2:
                        delta = torch.mm(up_weight, down_weight)
                    else:
                        # For conv layers
                        delta = torch.einsum('o i ..., i j ... -> o j ...', up_weight, down_weight)
                        
                    # Calculate scaling
                    dim = down_weight.shape[0] if len(down_weight.shape) == 2 else down_weight.shape[1]
                    lora_alpha = lora_parts.get('alpha', float(dim))
                    scale = alpha * (float(lora_alpha) / float(dim))
                    
                    original_weight = merged_sd[target_key].to(torch.float32)
                    merged_weight = original_weight + (delta.to(original_weight.device) * scale)
                    merged_sd[target_key] = merged_weight.to(merged_sd[target_key].dtype)
                    applied_count += 1
                else:
                    if target_key not in merged_sd:
                        print(f"Could not find matching base key in checkpoint for: {target_key}")

            print(f"Applied {applied_count} LoRA modules.")

        target_dtype = None
        if output_dtype == "fp16": target_dtype = torch.float16
        elif output_dtype == "bf16": target_dtype = torch.bfloat16
        elif output_dtype == "fp32": target_dtype = torch.float32
        
        if target_dtype is not None:
            progress(0.80, desc=f"Converting Precision to {output_dtype}...")
            print(f"Converting merged weights to {output_dtype}...")
            for k in merged_sd.keys():
                merged_sd[k] = merged_sd[k].to(target_dtype)

        progress(0.85, desc="Writing to Disk (This takes awhile)...")
        print(f"Saving to {output_path}...")
        
        # Ensure all tensors are contiguous before saving (this can prevent hangs on large models)
        for k in merged_sd.keys():
            merged_sd[k] = merged_sd[k].contiguous()
            
        save_file(merged_sd, output_path)
        progress(1.0, desc="Finished!")
        print("======== FINISHED ========")
        print(f"Merge successful! Saved to {output_path}")
        return f"Successfully merged and saved to {output_path}\nMerge Type: {merge_type}\nOutput Precision: {output_dtype}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error during merge: {str(e)}"

# Define Gradio UI with UltraMuse styling
custom_theme = gr.themes.Default(
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0b0f19",
    body_text_color="#f8fafc",
    block_background_fill="#1e293b",
    block_border_width="1px",
    block_border_color="#334155",
    block_title_text_color="#f8fafc",
    block_label_text_color="#cbd5e1",
    block_radius="12px",
    input_background_fill="#0f172a",
    input_border_color="#475569",
    input_border_color_focus="#6366f1",
    input_background_fill_focus="#0f172a",
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#334155",
    button_secondary_background_fill_hover="#475569",
    button_secondary_text_color="#f8fafc",
    slider_color="#6366f1",
    border_color_primary="#334155",
    checkbox_background_color="#0f172a",
    checkbox_label_background_fill="#0f172a",
    checkbox_label_background_fill_hover="#1e293b",
    checkbox_label_background_fill_selected="#4f46e5",
    checkbox_label_text_color="#f8fafc",
    checkbox_label_text_color_selected="#ffffff",
    checkbox_border_color="#475569",
    checkbox_border_color_focus="#6366f1",
)

custom_css = """
body {
    background-color: #0b0f19;
}
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}
h1 {
    color: #ffffff !important;
    text-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
}
p {
    color: #94a3b8 !important;
}
.gr-button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: background-color 0.2s, transform 0.1s !important;
}
.gr-button:active {
    transform: scale(0.98) !important;
}
"""

with gr.Blocks(title="Model Lora Merge tool by UltraMuse", theme=custom_theme, css=custom_css) as demo:
    gr.Markdown(
        """
        <div style='text-align: center; padding: 40px 20px 20px 20px;'>
            <h1 style='margin-bottom: 5px; font-weight: 800; letter-spacing: -0.02em; font-size: 3rem;'>UltraMuse</h1>
            <p style='font-size: 1.1em; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase;'>Model & LoRA Integration Engine</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### ⚙️ Configuration")
                merge_type = gr.Radio(
                    ["Checkpoint + Checkpoint (Weighted Sum)", "Checkpoint + LoRA"], 
                    label="Merge Strategy", 
                    value="Checkpoint + Checkpoint (Weighted Sum)"
                )
                
                with gr.Row():
                    model_a = gr.Textbox(label="Model A (Base Checkpoint)", placeholder="C:/path/to/base.safetensors", scale=4)
                    pick_a_btn = gr.Button("📁 Open", scale=1, size="sm")

                with gr.Row():
                    model_b = gr.Textbox(label="Model B / LoRA Checkpoint", placeholder="C:/path/to/model_B.safetensors", scale=4)
                    pick_b_btn = gr.Button("📁 Open", scale=1, size="sm")

            with gr.Group():
                gr.Markdown("### 🎛️ Blending Options")
                alpha = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Alpha Strength (Ratio / LoRA Weight)")
                gr.Markdown("<span style='color: #94a3b8; font-size: 0.85em;'>*0.5 splits checkpoints equally. 1.0 applies 100% of a LoRA.*</span>")

                output_dtype = gr.Dropdown(
                    ["Keep Original", "fp32", "fp16", "bf16"], 
                    label="Output Precision", 
                    value="Keep Original"
                )

            with gr.Group():
                gr.Markdown("### 💾 Output Destination")
                with gr.Row():
                    output_path = gr.Textbox(label="Save Path", value="output/merged.safetensors", scale=4)
                    pick_out_btn = gr.Button("📁 Directory", scale=1, size="sm")
                    
            merge_btn = gr.Button("🚀 Execute Merge", variant="primary", size="lg")

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📊 Status Engine")
                output_log = gr.Textbox(label="Console Output", lines=15, interactive=False, container=False)

    # Event Bindings
    pick_a_btn.click(fn=pick_file, outputs=model_a)
    pick_b_btn.click(fn=pick_file, outputs=model_b)
    pick_out_btn.click(fn=pick_out_dir, outputs=output_path)
            
    merge_btn.click(
        fn=merge_models,
        inputs=[model_a, model_b, merge_type, alpha, output_path, output_dtype],
        outputs=output_log
    )

if __name__ == "__main__":
    print("Starting UltraMuse Model Merger on http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)

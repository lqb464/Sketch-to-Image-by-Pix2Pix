import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.networks import define_G

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path):
    """Load U-Net model from checkpoint"""
    try:
        netG = define_G(
            input_nc=3,      # sketch (grayscale)
            output_nc=3,     # rgb image
            ngf=64,
            netG='unet_256',
            norm='batch',
            use_dropout=False,
            init_type='normal',
            init_gain=0.02
        ).to(device)
        
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(state_dict, dict):
            if 'netG' in state_dict:
                netG.load_state_dict(state_dict['netG'])
            else:
                netG.load_state_dict(state_dict)
        else:
            netG.load_state_dict(state_dict)
        
        netG.eval()
        return netG
        
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return None

# Load model from available paths
model = None
checkpoint_paths = [
    "best_model.pth",
    "results/checkpoints/best_model.pth",
    "checkpoints/best_model.pth"
]

for path in checkpoint_paths:
    if Path(path).exists():
        try:
            model = load_model(path)
            if model is not None:
                print(f"[INFO] Model loaded successfully from: {path}")
                break
        except Exception as e:
            continue

if model is None:
    print("[WARNING] No checkpoint found! Inference will fail.")

def preprocess_image(image, size=256):
    """Convert PIL image to normalized tensor (RGB)"""
    if image is None:
        raise ValueError("No image provided")
    
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_np = np.array(image, dtype=np.float32) / 255.0
    img_np = img_np * 2 - 1
    img_np = np.transpose(img_np, (2, 0, 1))
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)
    
    return img_tensor

def sketch_to_image(sketch_image):
    """Main inference function"""
    if model is None:
        return None, "Model not loaded. Please check checkpoint path."
    
    if sketch_image is None:
        return None, "Please upload a sketch image first."
    
    try:
        sketch_tensor = preprocess_image(sketch_image).to(device)
        
        with torch.no_grad():
            output = model(sketch_tensor)
        
        output_np = output[0].cpu().clamp(-1, 1).permute(1, 2, 0).numpy()
        output_np = ((output_np + 1) / 2 * 255).astype(np.uint8)
        output_image = Image.fromarray(output_np)
        
        return output_image, "Generation successful!"
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Sketch to Image") as demo:
    gr.Markdown("""
    # Sketch to Image Generator
    Convert your sketch drawings into realistic images using AI!
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input: Sketch")
            sketch_input = gr.Image(
                type="pil",
                label="Upload Sketch",
                interactive=True,
                scale=1
            )
        
        with gr.Column():
            gr.Markdown("### Output: Generated Image")
            image_output = gr.Image(
                type="pil",
                label="Generated Image",
                interactive=False,
                scale=1
            )
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                value="Upload a sketch and click Generate"
            )
    
    generate_button = gr.Button("Generate", variant="primary", scale=2)
    generate_button.click(
        fn=sketch_to_image,
        inputs=sketch_input,
        outputs=[image_output, status_output]
    )
    
    gr.Markdown("""
    ---
    ### Tips
    - Use clear, black sketch on white background for best results
    - Images will be automatically resized to 256×256 pixels
    - Inference time depends on your hardware (GPU ~0.5s, CPU ~5s)
    
    ### Model Info
    - **Architecture**: U-Net Generator with 256×256 resolution
    - **Input**: Grayscale sketch (1 channel)
    - **Output**: RGB realistic image (3 channels)
    - **Framework**: PyTorch + Pix2Pix
    """)

if __name__ == "__main__":
    demo.launch(share=True, server_name="127.0.0.1", server_port=7860)
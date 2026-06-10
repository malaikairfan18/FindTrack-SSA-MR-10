import os
import subprocess
import sys

def setup():
    print("=== Installing dependencies ===")
    packages = [
        "loralib",
        "ftfy",
        "omegaconf",
        "hydra-core",
        "torchscale",
        "simpletransformers",
        "accelerate",
        "gradio",
        "imageio",
        "timm"
    ]
    for pkg in packages:
        print(f"Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg])

    print("=== Installing gdown ===")
    subprocess.run([sys.executable, "-m", "pip", "install", "gdown"])

    print("=== Creating weights/ directory ===")
    os.makedirs("weights", exist_ok=True)

    print("=== Downloading Alpha-CLIP weights via gdown ===")
    file_id = "1dG_j98hh7AFvhSADlhp9CpoNY-9rBHoc"
    output_path = "weights/clip_l14_336_grit_20m_4xe.pth"
    
    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping download.")
    else:
        import gdown
        gdown.download(id=file_id, output=output_path, quiet=False)
        print("=== Alpha-CLIP weights downloaded successfully ===")
        
    print("=== Setup Completed successfully! ===")

if __name__ == '__main__':
    setup()

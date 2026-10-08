
# Prototype1.py: For users with Python installed
# This script will install all requirements and run the app.

import subprocess
import sys
import os

# (pip name, import name)
requirements = [
    ("customtkinter", "customtkinter"),
    ("Pillow", "PIL"),
    ("numpy", "numpy"),
    ("opencv-python-headless", "cv2"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("timm", "timm"),
    ("scikit-learn", "sklearn"),
]

def install(package):
    pip_name, module = package
    try:
        __import__(module)
    except ImportError:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

def has_nvidia_gpu():
    import shutil
    return shutil.which("nvidia-smi") is not None and subprocess.call(
        ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


# On machines with an NVIDIA GPU install the CUDA build of PyTorch (the default pip wheel is CPU-only
# on Windows); the tool then uses the GPU automatically.
try:
    import torch
    need_cuda = has_nvidia_gpu() and not torch.cuda.is_available()
except ImportError:
    need_cuda = has_nvidia_gpu()
if need_cuda:
    print("NVIDIA GPU found - installing CUDA-enabled PyTorch...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "torch", "torchvision",
                           "--index-url", "https://download.pytorch.org/whl/cu130"])

for pkg in requirements:
    install(pkg)

print("\nAll requirements installed. Launching the app...")
os.system(f'"{sys.executable}" Interface.py')

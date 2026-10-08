
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

for pkg in requirements:
    install(pkg)

print("\nAll requirements installed. Launching the app...")
os.system(f'"{sys.executable}" Interface.py')

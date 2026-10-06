
# Prototype1.py: For users with Python installed
# This script will install all requirements and run the app.

import subprocess
import sys
import os

requirements = [
    "customtkinter",
    "Pillow",
    "torch",
    "torchvision",
    "matplotlib",
    "numpy"
]

def install(package):
    try:
        __import__(package.split('==')[0])
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in requirements:
    install(pkg)

print("\nAll requirements installed. Launching the app...")
os.system(f'"{sys.executable}" Interface.py')

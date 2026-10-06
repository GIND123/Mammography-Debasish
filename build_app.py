import os
import sys
import subprocess
import shutil

def install_requirements():
    print("Installing required packages...")
    requirements = [
        "customtkinter",
        "pillow",
        "torch",
        "torchvision",
        "matplotlib",
        "numpy",
        "pyinstaller"
    ]
    for package in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def create_spec_content():
    return '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['Interface.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('Background.png', '.'),
        ('bgcolour.png', '.'),
        ('Model.py', '.'),
        ('best_model_final.pth', '.'),
        ('full_model.pt', '.'),
        ('CC Image.jpg', '.'),
        ('MLOimage.jpg', '.')
    ],
    hiddenimports=[
        'customtkinter',
        'torch',
        'torchvision',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'matplotlib',
        'numpy'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Interface',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None
)
'''

def create_inno_setup_script():
    return '''[Setup]
AppName=DiceMed
AppVersion=1.0
DefaultDirName={autopf}\\DiceMed
DefaultGroupName=DiceMed
OutputBaseFilename=DiceMedSetup
Compression=lzma
SolidCompression=yes
DisableDirPage=auto

[Files]
Source: "dist\\Interface.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "Background.png"; DestDir: "{app}"; Flags: ignoreversion
Source: "bgcolour.png"; DestDir: "{app}"; Flags: ignoreversion
Source: "Model.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "best_model_final.pth"; DestDir: "{app}"; Flags: ignoreversion
Source: "full_model.pt"; DestDir: "{app}"; Flags: ignoreversion
Source: "CC Image.jpg"; DestDir: "{app}"; Flags: ignoreversion
Source: "MLOimage.jpg"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\\DiceMed"; Filename: "{app}\\DiceMed.exe"
Name: "{commondesktop}\\DiceMed"; Filename: "{app}\\DiceMed.exe"
'''

def main():
    # Clean previous builds
    print("Cleaning previous builds...")
    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    
    # Install requirements
    install_requirements()
    
    # Create spec file
    print("Creating spec file...")
    with open('DiceMed.spec', 'w') as f:
        f.write(create_spec_content())
    
    # Build executable
    print("Building executable...")
    subprocess.check_call([sys.executable, '-m', 'PyInstaller', 'DiceMed.spec', '--clean'])
    
    # Create Inno Setup script
    print("Creating Inno Setup script...")
    with open('installer.iss', 'w') as f:
        f.write(create_inno_setup_script())
    
    print("\nBuild complete!")
    print("1. Install Inno Setup from: https://jrsoftware.org/isdl.php")
    print("2. Open installer.iss with Inno Setup Compiler")
    print("3. Click Compile to create the installer")
    print("\nThe final installer will be in the Output folder.")

if __name__ == "__main__":
    main()

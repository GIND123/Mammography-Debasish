# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Add all submodules
hidden_imports = [
    'customtkinter',
    'tkinter',
    'PIL',
    'torch',
    'torchvision',
    'matplotlib',
    'numpy',
    'io',
    'sys',
    'importlib',
    'os'
] + collect_submodules('torch') + collect_submodules('torchvision')

# Add all data files
datas = [
    ('Background.png', '.'),
    ('bgcolour.png', '.'),
    ('CC Image.jpg', '.'),
    ('MLOimage.jpg', '.'),
    ('best_model_final.pth', '.'),
    ('full_model.pt', '.'),
    ('Model.py', '.')
]

# Add customtkinter data files
datas += collect_data_files('customtkinter')

a = Analysis(
    ['Interface.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
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

; DiceMed Inno Setup Script
; Save this as installer.iss and open with Inno Setup Compiler

[Setup]
AppName=DiceMed
AppVersion=1.0
DefaultDirName={pf}\DiceMed
DefaultGroupName=DiceMed
OutputBaseFilename=DiceMedInstaller
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\Interface.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "Background.png"; DestDir: "{app}"; Flags: ignoreversion
Source: "bgcolour.png"; DestDir: "{app}"; Flags: ignoreversion
Source: "Model.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "best_model_final.pth"; DestDir: "{app}"; Flags: ignoreversion
Source: "full_model.pt"; DestDir: "{app}"; Flags: ignoreversion
Source: "CC Image.jpg"; DestDir: "{app}"; Flags: ignoreversion
Source: "MLOimage.jpg"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\DiceMed"; Filename: "{app}\Interface.exe"
Name: "{userdesktop}\DiceMed"; Filename: "{app}\Interface.exe"

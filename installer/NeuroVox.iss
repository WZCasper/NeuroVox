; Установщик NeuroVox (Inno Setup 6). Собирается автоматически в GitHub Actions.
; Ручная сборка: ISCC.exe /DAppVersion=1.0.0 installer\NeuroVox.iss  (после pyinstaller NeuroVox.spec)

#define AppName "NeuroVox"
#define AppExe "NeuroVox.exe"
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{B0C6C1E2-6A1F-4E0B-9D3F-4E5B7A2C9F11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=WZCasper
AppPublisherURL=https://github.com/WZCasper/NeuroVox
; Пути ниже считаются от корня репозитория.
SourceDir=..
OutputDir=dist-installer
OutputBaseFilename=NeuroVox-Setup
SetupIconFile=assets\neurovox.ico
UninstallDisplayIcon={app}\{#AppExe}
; Установка «для текущего пользователя»: права администратора не нужны.
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
ShowLanguageDialog=no

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; GroupDescription: "Дополнительно:"

[Files]
Source: "dist\NeuroVox\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Запустить NeuroVox"; Flags: nowait postinstall skipifsilent

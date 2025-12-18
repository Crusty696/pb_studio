; ============================================================================
; Inno Setup Script für PB_studio Windows Installer
; ============================================================================
; Erstellt Windows Installer (.exe) aus PyInstaller Build
;
; Voraussetzungen:
;   - Inno Setup 6.x installiert (https://jrsoftware.org/isinfo.php)
;   - Erfolgreicher PyInstaller Build (dist/PB_studio/)
;
; Build:
;   iscc scripts/installer.iss
;
; Output:
;   Output/PB_studio_Setup_v1.0.0.exe
;
; Author: PB_studio Development Team
; Task: D4 - Inno Setup Installer
; ============================================================================

[Setup]
; ============================================================================
; Basic App Information
; ============================================================================
AppName=PB_studio
AppVersion=1.0.0
AppPublisher=PB_studio Development Team
AppPublisherURL=https://github.com/yourrepo/pb_studio
AppSupportURL=https://github.com/yourrepo/pb_studio/issues
AppUpdatesURL=https://github.com/yourrepo/pb_studio/releases
AppCopyright=Copyright (C) 2025 PB_studio Team

; ============================================================================
; Installation Directories
; ============================================================================
DefaultDirName={autopf}\PB_studio
DefaultGroupName=PB_studio
DisableProgramGroupPage=yes

; ============================================================================
; Output Configuration
; ============================================================================
OutputDir=Output
OutputBaseFilename=PB_studio_Setup_v1.0.0
SetupIconFile=..\src\pb_studio\resources\icons\app_icon.ico
UninstallDisplayIcon={app}\PB_studio.exe

; ============================================================================
; Compression & Installer Options
; ============================================================================
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMADictionarySize=1048576
LZMANumBlockThreads=2

; ============================================================================
; Visual Style
; ============================================================================
WizardStyle=modern
WizardImageFile=compiler:WizModernImage-IS.bmp
WizardSmallImageFile=compiler:WizModernSmallImage-IS.bmp

; ============================================================================
; Privileges & Compatibility
; ============================================================================
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
MinVersion=10.0.14393
ArchitecturesInstallIn64BitMode=x64

; ============================================================================
; Uninstaller
; ============================================================================
UninstallDisplayName=PB_studio
UninstallFilesDir={app}\uninstall

; ============================================================================
; License & Info Files
; ============================================================================
; LicenseFile=..\LICENSE.txt
; InfoBeforeFile=..\README.md
; InfoAfterFile=..\CHANGELOG.md

; ============================================================================
; Languages
; ============================================================================
[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"

; ============================================================================
; Tasks (Optional Features)
; ============================================================================
[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

; ============================================================================
; Files to Install
; ============================================================================
[Files]
; Main Application
Source: "..\dist\PB_studio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Additional Resources (if needed)
; Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
; Source: "..\LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion

; ============================================================================
; Icons & Shortcuts
; ============================================================================
[Icons]
; Start Menu Entry
Name: "{group}\PB_studio"; Filename: "{app}\PB_studio.exe"; WorkingDir: "{app}"; Comment: "Video Editing mit Beat-Matching"

; Desktop Icon (optional)
Name: "{autodesktop}\PB_studio"; Filename: "{app}\PB_studio.exe"; WorkingDir: "{app}"; Tasks: desktopicon

; Quick Launch Icon (optional, legacy)
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\PB_studio"; Filename: "{app}\PB_studio.exe"; WorkingDir: "{app}"; Tasks: quicklaunchicon

; Uninstaller
Name: "{group}\Uninstall PB_studio"; Filename: "{uninstallexe}"

; ============================================================================
; Registry Entries (Optional - File Associations)
; ============================================================================
[Registry]
; Register application
Root: HKCU; Subkey: "Software\PB_studio"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\PB_studio"; ValueType: string; ValueName: "Version"; ValueData: "1.0.0"
Root: HKCU; Subkey: "Software\PB_studio"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"

; File Association für .pbproj (PB_studio Project Files)
; Root: HKCR; Subkey: ".pbproj"; ValueType: string; ValueName: ""; ValueData: "PB_studio.Project"; Flags: uninsdeletevalue
; Root: HKCR; Subkey: "PB_studio.Project"; ValueType: string; ValueName: ""; ValueData: "PB_studio Project"; Flags: uninsdeletekey
; Root: HKCR; Subkey: "PB_studio.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\PB_studio.exe,0"
; Root: HKCR; Subkey: "PB_studio.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\PB_studio.exe"" ""%1"""

; ============================================================================
; Run After Installation
; ============================================================================
[Run]
; Option to launch application after install
Filename: "{app}\PB_studio.exe"; Description: "{cm:LaunchProgram,PB_studio}"; Flags: nowait postinstall skipifsilent

; ============================================================================
; Custom Messages
; ============================================================================
[Messages]
; English
english.WelcomeLabel1=Welcome to [name] Setup
english.WelcomeLabel2=This will install [name/ver] on your computer.%n%nPB_studio is a professional video editing tool with automatic beat-matching and AI-powered pacing.
english.FinishedLabel=Setup has finished installing [name] on your computer. The application may be launched by selecting the installed icons.

; German
german.WelcomeLabel1=Willkommen beim [name] Setup
german.WelcomeLabel2=Dieses Programm wird [name/ver] auf Ihrem Computer installieren.%n%nPB_studio ist ein professionelles Video-Editing-Tool mit automatischem Beat-Matching und KI-gestütztem Pacing.
german.FinishedLabel=Das Setup hat [name] erfolgreich auf Ihrem Computer installiert. Die Anwendung kann über die installierten Symbole gestartet werden.

; ============================================================================
; Code Section (Pascal Script)
; ============================================================================
[Code]
// ============================================================================
// Custom Functions
// ============================================================================

// Check if .NET Framework 4.8+ is installed (if needed)
function IsDotNetInstalled: Boolean;
var
  exists: Boolean;
  release: Cardinal;
begin
  // Check for .NET Framework 4.8+ (Release >= 528040)
  exists := RegQueryDWordValue(HKLM, 'SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full', 'Release', release);
  Result := exists and (release >= 528040);
end;

// Custom page for additional options (if needed)
{
procedure InitializeWizard;
begin
  // Add custom wizard pages here if needed
end;
}

// Check prerequisites before installation
function InitializeSetup: Boolean;
begin
  Result := True;

  // Example: Check for required components
  {
  if not IsDotNetInstalled then
  begin
    MsgBox('.NET Framework 4.8 or higher is required but not installed.' + #13#10 +
           'Please install .NET Framework 4.8 first.', mbError, MB_OK);
    Result := False;
  end;
  }
end;

// Actions after successful installation
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Create application data directory for user settings
    ForceDirectories(ExpandConstant('{userappdata}\PB_studio'));
  end;
end;

// Custom uninstall actions
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  UserDataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    // Ask user if they want to remove user data
    if MsgBox('Do you want to remove all PB_studio user data and settings?' + #13#10 +
              'This includes all projects and preferences.', mbConfirmation, MB_YESNO) = IDYES then
    begin
      UserDataDir := ExpandConstant('{userappdata}\PB_studio');
      if DirExists(UserDataDir) then
      begin
        DelTree(UserDataDir, True, True, True);
      end;
    end;
  end;
end;

// ============================================================================
// End of Inno Setup Script
// ============================================================================

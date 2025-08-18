@echo off
echo 🎬 UpScale App - Building Executable
echo =======================================

REM Change to project directory
cd /d "%~dp0"

echo 📦 Installing build dependencies...
python -m pip install --upgrade pip
python -m pip install -r build_requirements.txt

echo 🔨 Building executable with PyInstaller...
python -m PyInstaller upscale_app.spec --clean --noconfirm

echo ✅ Build process completed!
echo 📁 Executable location: dist\UpScaleApp.exe

if exist "dist\UpScaleApp.exe" (
    echo 🎉 Build successful! 
    echo 📊 File size:
    dir "dist\UpScaleApp.exe" | findstr "UpScaleApp.exe"
) else (
    echo ❌ Build failed - executable not found
)

pause
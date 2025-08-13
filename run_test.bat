@echo off
echo 🎬 UpScale App - Windows Environment Test
echo.

REM Try different Python commands
echo 🔍 Testing Python availability...

python --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ python command available
    python quick_test.py
    goto :end
)

python3 --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ python3 command available  
    python3 quick_test.py
    goto :end
)

py --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ py launcher available
    py quick_test.py
    goto :end
)

echo ❌ No Python installation found
echo.
echo 📝 Please install Python from https://python.org
echo    Make sure to check "Add Python to PATH" during installation
echo.
echo 🔧 Alternative: Try running manually:
echo    C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python39\python.exe quick_test.py
pause

:end
echo.
pause
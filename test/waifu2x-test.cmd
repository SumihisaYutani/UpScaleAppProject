@echo off
rem Waifu2x GPU Optimization Test Script
rem Equivalent to BLOCKSIZE optimization mentioned in the reference

set WAIFU2X_PATH=..\tools\waifu2x-ncnn-vulkan\waifu2x-ncnn-vulkan-20220728-windows\waifu2x-ncnn-vulkan.exe
set INPUT_IMAGE=%1
set OUTPUT_IMAGE=test_result.png

rem BLOCKSIZE equivalent: tile size (-t parameter)
rem Start with 256, test with 512, 1024, 2048 etc.
rem Larger values = faster processing but more GPU memory usage
set TILE_SIZE=512

rem GPU ID: 0 = Radeon RX Vega, 1 = Intel HD Graphics
set GPU_ID=0

rem Threading: load:proc:save
set THREADS=1:4:2

echo Testing Waifu2x with Radeon RX Vega optimization...
echo Tile Size (BLOCKSIZE): %TILE_SIZE%
echo GPU ID: %GPU_ID%
echo Input: %INPUT_IMAGE%
echo Output: %OUTPUT_IMAGE%
echo.

"%WAIFU2X_PATH%" -i "%INPUT_IMAGE%" -o "%OUTPUT_IMAGE%" -s 2 -n 1 -g %GPU_ID% -t %TILE_SIZE% -j %THREADS% -m models-cunet -f png -v

if exist "%OUTPUT_IMAGE%" (
    echo.
    echo SUCCESS: Test image processed successfully!
    echo Output saved as: %OUTPUT_IMAGE%
    echo.
    echo Try increasing TILE_SIZE for better performance:
    echo - 256 (conservative)
    echo - 512 (current)  
    echo - 1024 (high performance)
    echo - 2048 (maximum, may cause errors)
) else (
    echo.
    echo ERROR: Processing failed!
    echo Try reducing TILE_SIZE value or check GPU drivers
)

pause
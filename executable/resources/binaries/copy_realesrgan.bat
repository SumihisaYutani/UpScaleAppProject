@echo off
echo Copying Real-ESRGAN files...
copy "realesrgan-temp\realesrgan-ncnn-vulkan.exe" .
xcopy /s /y "realesrgan-temp\models" "realesrgan-models\"
echo Done!
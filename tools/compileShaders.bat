@echo off
REM Compile shaders that are loaded as pre-built SPIR-V at runtime.
REM Search and ray-march shaders are generated/compiled at runtime via Slang
REM and don't need offline compilation.
REM
REM Requires the Vulkan SDK installer to have set %VULKAN_SDK%.

if "%VULKAN_SDK%"=="" (
	echo ERROR: VULKAN_SDK environment variable not set. Install Vulkan SDK and re-run.
	exit /b 1
)

"%VULKAN_SDK%\Bin\glslc.exe" "%~dp0..\shaders\source\encoding\rgb2ycbcr.comp" -o "%~dp0..\shaders\compiled\rgb2ycbcr.spv"
if errorlevel 1 exit /b 1

echo Shader compilation complete.

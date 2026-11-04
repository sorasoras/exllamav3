@echo off
rem Sweep one int8 GEMM variant without a full extension rebuild.
rem   tune.bat <variant-name>    (knobs come from tune_defs.h)
rem Prints ms, TMAC/s and the percentage of the measured dp4a peak for 9B layer shapes.
setlocal
cd /d "%~dp0"
if "%CUDA_HOME%"=="" set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set VSW_HERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
for /f "usebackq tokens=*" %%i in (`"%VSW_HERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
"%CUDA_HOME%\bin\nvcc.exe" -arch=sm_61 -O3 -Xptxas -v -include tune_defs.h -o tune_%1.exe sm61_i8_tune.cu > tune_%1.log 2>&1
if errorlevel 1 (echo COMPILE_FAIL & type tune_%1.log & exit /b 1)
tune_%1.exe

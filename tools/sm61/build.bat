@echo off
rem Build the exllamav3 CUDA extension for a Pascal GPU (sm_61).
rem
rem nvcc must see an MSVC that PyTorch accepts. The Visual Studio developer environment is
rem entered explicitly so an unrelated newer cl.exe on PATH cannot be picked up instead.
rem Override CUDA_HOME or TORCH_CUDA_ARCH to build for a different toolkit / card.
setlocal
cd /d "%~dp0..\.."

if "%CUDA_HOME%"=="" set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
if "%TORCH_CUDA_ARCH%"=="" set TORCH_CUDA_ARCH=6.1

set VSW_HERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
if not exist "%VSW_HERE%" (
    echo vswhere.exe not found - install Visual Studio with the C++ build tools
    exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%VSW_HERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSPATH=%%i
if "%VSPATH%"=="" (
    echo No Visual Studio installation with the C++ build tools found
    exit /b 1
)
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

set TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH%
set MAX_JOBS=6
set DISTUTILS_USE_SDK=1
set MSSdk=1

rem An editable install is assumed (pip install -e .), so this only rebuilds the extension.
python -m pip install -e . --no-build-isolation --no-deps > build_sm61.log 2>&1
set RC=%ERRORLEVEL%
echo BUILD_EXIT=%RC% >> build_sm61.log
if not "%RC%"=="0" type build_sm61.log
exit /b %RC%

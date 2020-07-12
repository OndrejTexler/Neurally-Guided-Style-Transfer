@echo off
setlocal ENABLEDELAYEDEXPANSION

call "vcvarsall.bat" amd64

nvcc -arch compute_30 SynthesisUtils.cpp EbsynthWrapper.cpp main.cpp OpenCVUtils.cpp patch_based_synthesis\src\ebsynth.cpp patch_based_synthesis\src\ebsynth_cpu.cpp patch_based_synthesis\src\ebsynth_cuda.cu -DNDEBUG -O6 -I "opencv-4.2.0\include" -I "patch_based_synthesis\include" -o styletransfer.exe -Xcompiler "/openmp /EHsc /nologo" -Xlinker "/LIBPATH:"opencv-4.2.0\lib" opencv_world420.lib" || goto error

pause
goto :EOF

:error
echo FAILED
@%COMSPEC% /C exit 1 >nul
pause
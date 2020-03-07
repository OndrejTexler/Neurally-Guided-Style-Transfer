@echo off
setlocal ENABLEDELAYEDEXPANSION

call "vcvarsall.bat" amd64

cl CreateGuideUtils.cpp EbsynthWrapper.cpp main.cpp OpenCVUtils.cpp patch_based_synthesis\src\ebsynth.cpp patch_based_synthesis\src\ebsynth_cpu.cpp patch_based_synthesis\src\ebsynth_nocuda.cpp /DNDEBUG /O2 /openmp /EHsc /nologo /I"opencv-4.2.0\include" /I"patch_based_synthesis\include" /Fe"styletransfer.exe" /link /LIBPATH:"opencv-4.2.0\lib" opencv_world420.lib || goto error

del CreateGuideUtils.obj;EbsynthWrapper.obj;main.obj;OpenCVUtils.obj;ebsynth.obj;ebsynth_cpu.obj;ebsynth_nocuda.obj 2> NUL
pause
goto :EOF

:error
echo FAILED
@%COMSPEC% /C exit 1 >nul
pause
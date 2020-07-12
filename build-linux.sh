#!/bin/sh
g++ SynthesisUtils.cpp \
	EbsynthWrapper.cpp \
	main.cpp \
	OpenCVUtils.cpp \
	patch_based_synthesis/src/ebsynth.cpp \
	patch_based_synthesis/src/ebsynth_cpu.cpp \
	patch_based_synthesis/src/ebsynth_nocuda.cpp \
	-DNDEBUG \
	-O6 \
	-fopenmp \
	-I"opencv-4.2.0/include" \
	-I"patch_based_synthesis/include" \
	-std=c++11 \
	-o styletransfer \
	-L"opencv-4.2.0/lib" \
	-lopencv_world

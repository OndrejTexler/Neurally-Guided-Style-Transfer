#!/bin/sh
nvcc -arch compute_30 \
	SynthesisUtils.cpp \
	EbsynthWrapper.cpp \
	main.cpp \
	OpenCVUtils.cpp \
	patch_based_synthesis/src/ebsynth.cpp \
	patch_based_synthesis/src/ebsynth_cpu.cpp \
	patch_based_synthesis/src/ebsynth_cuda.cu \
	-I"opencv-4.2.0/include" \
	-I"patch_based_synthesis/include" \
	-DNDEBUG \
	-O6 \
	-std=c++11 \
	-w \
	-o styletransfer \
	-Xcompiler \
	-fopenmp \
	-Xlinker \
	-L"opencv-4.2.0/lib" \
	-lopencv_world
	
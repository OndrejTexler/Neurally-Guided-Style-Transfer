#!/bin/sh
nvcc -arch compute_30 \
	src/ebsynth.cpp \
	src/ebsynth_cpu.cpp \
	src/ebsynth_cuda.cu \
	-I"include" \
	-DNDEBUG \
	-D__CORRECT_ISO_CPP11_MATH_H_PROTO \
	-D_MWAITXINTRIN_H_INCLUDED \
	-D_FORCE_INLINES \
	-O6 \
	-std=c++11 \
	-w \
	-Xcompiler \
	-fopenmp \
	-o bin/ebsynth

nvcc -arch compute_30 \
	src/ebsynth.cpp \
	src/ebsynth_cpu.cpp \
	src/ebsynth_cuda.cu \
	-I"include" \
	-DNDEBUG \
	-D__CORRECT_ISO_CPP11_MATH_H_PROTO \
	-D_MWAITXINTRIN_H_INCLUDED \
	-D_FORCE_INLINES \
	-O6 \
	-std=c++11 \
	-w \
	-Xcompiler -fopenmp \
	-Xcompiler -fPIC \
	-shared \
	-o lib/libebsynth.so
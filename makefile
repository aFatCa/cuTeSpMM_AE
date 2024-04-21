
# CUSPARSELT_DIR	:= $(HOME)/opt/libcusparse_lt
# CUDA_TOOLKIT := $(HOME)/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/math_libs
# INC          := -I$(CUDA_TOOLKIT)/include 
# LIBS         := -L$(CUDA_TOOLKIT)/lib64 
INCCUSPLT	 := -I$(CUSPARSELT_DIR)/include
LIBSCUSPLT	 := -L$(CUSPARSELT_DIR)/include
INCCUTLASS	 := -I$(HOME)/opt/cutlass/include

INCLIKWID	 := -I/uufs/chpc.utah.edu/sys/installdir/likwid/5.0.1/include
LIBLIKWID	 := -L/uufs/chpc.utah.edu/sys/installdir/likwid/5.0.1/lib

INCSPUTNIK	 := -I$(HOME)/opt/sputnik/
LIBSPUTNIK	 := -L$(HOME)/opt/sputnik/build/sputnik


INCABSL	 := -I$(HOME)/opt/sputnik/third_party/abseil-cpp
LIBABSL	 := -L$(HOME)/optsputnik/third_party/abseil-cpp

PROJ_DIR=$(HOME)/opt/dgSPARSE-Lib/src/ge-spmm
LIB = $(PROJ_DIR)/libgespmm.a
SO = $(HOME)/opt/dgSPARSE-Lib/lib/dgsparse.so

INCGFLG	 := -I$(HOME)/opt/gflags/build/include
LIBGFLG	 := -L$(HOME)/opt/gflags/build/lib

all: all_spmm
.PHONY : clean 


gpu_kernels:
	nvcc -O3 -std=c++14 -I$(PROJ_DIR) $(INCABSL) $(INCSPUTNIK) $(INCGFLG) $(LIBGFLG) gpu_kernels.cu -o gpu_kernels.o  $(LIB) $(LIBABSL) $(LIBSPUTNIK) -lsputnik -lglog -lcudart -lcusparse -lcublas -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -lnvToolsExt -DCOMPUTETYPE=${ComputeType}

clean:
	rm -rf *.o core.* *.log *.err raw/* *cudafe* *fatbin* *dlink* *ii *module_id *cubin *.ptx

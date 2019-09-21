NVCC=nvcc -Wno-deprecated-gpu-targets
OBJ_FLAGS=-x cu -I. -dc
LIB_FLAGS=-shared
GCC_FLAGS=-Xcompiler -fPIC

.PONY: all
all:
	@mkdir -p ../lib/obj/
	@${NVCC} ${OBJ_FLAGS} cuDeviceProperties.cu -o ../lib/obj/cuDeviceProperties.o ${GCC_FLAGS}
	@${NVCC} ${LIB_FLAGS} -o ../lib/barracuda.so `ls ../lib/obj/*.o` ${GCC_FLAGS}

%.cu: ../lib/obj/%.o
	@mkdir -p ../lib/obj/
	@${NVCC} ${OBJ_FLAGS} $< -o $@ ${GCC_FLAGS}


.PHONY: clean
clean:
	@rm -rf ../lib/obj

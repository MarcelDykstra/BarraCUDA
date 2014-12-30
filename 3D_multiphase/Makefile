#------------------------------------------------------------------------------
export MATLAB      = /opt/Matlab
MEXSUFFIX   = ${shell ${MATLAB}/bin/mexext}
MEX         = ${MATLAB}/bin/mex
CXX         = g++
CFLAGS_NVCC = -fPIC,-D_GNU_SOURCE,-pthread,-fexceptions
CFLAGS      = -fPIC -D_GNU_SOURCE -pthread -fexceptions
COPTIMFLAGS = -O3 -funroll-loops -msse2

#------------------------------------------------------------------------------
CUDA      = /opt/cuda
INC_CUDA  = -I${CUDA}/include
LIB_CUDA  = -L${CUDA}/lib64
LIBS_CUDA = -lcuda -lcudart

#------------------------------------------------------------------------------
LIBS_MEX  = -lm
INC_MEX   = -I $(MATLAB)/extern/include
FLAGS_MEX = -cxx CC='$(CXX)' CXX='$(CXX)' LD='$(CXX)'

NVMEX = ../nvmex/nvmex -largeArrayDims
NVOPTS = ../nvmex/nvopts

#------------------------------------------------------------------------------
SRC = ./src
BIN = ./bin
LIB = ${SRC}/.libs
BACKUP   = ./backup
LOG_FILE = make.log
LOG_ERR  = 2>> ${LOG_FILE}
LOG_ECHO = | tee -a ${LOG_FILE}

#----------------------------------------------------------------------[all]---
.PHONY: all
all: init-make	${BIN}/cuLatticeInit.${MEXSUFFIX} \
		${BIN}/cuLatticeClear.${MEXSUFFIX} \
		${BIN}/cuCollidePropagate.${MEXSUFFIX} \
		${BIN}/cuDeviceProperties.${MEXSUFFIX} \
		${BIN}/cuDeviceReset.${MEXSUFFIX}
	@echo ===[ barracuda ]====== ${LOG_ECHO}

#---------------------------------------------------------[cuLatticeInit.cu]---
${BIN}/cuLatticeInit.${MEXSUFFIX}: ${SRC}/cuLatticeInit.cu \
		${SRC}/barracuda.cu ${SRC}/barracuda.h
	@echo ---[ ${SRC}/cuLatticeInit.cu ]------ ${LOG_ECHO}
	@$(NVMEX) -f $(NVOPTS) ${INC_CUDA} \
	 ${LIB_CUDA} ${LIBS_CUDA} \
	${SRC}/cuLatticeInit.cu ${SRC}/barracuda.cu \
	-o ${BIN}/cuLatticeInit ${LOG_ERR}

#--------------------------------------------------------[cuLatticeClear.cu]---
${BIN}/cuLatticeClear.${MEXSUFFIX}: ${SRC}/cuLatticeClear.cu \
		${SRC}/barracuda.cu ${SRC}/barracuda.h
	@echo ---[ ${SRC}/cuLatticeClear.cu ]------ ${LOG_ECHO}
	@$(NVMEX) -f $(NVOPTS) ${INC_CUDA} \
	 ${LIB_CUDA} ${LIBS_CUDA} \
	${SRC}/cuLatticeClear.cu ${SRC}/barracuda.cu \
	-o ${BIN}/cuLatticeClear ${LOG_ERR}

#---------------------------------------------------[cuCollidePropagate..cu]---
${BIN}/cuCollidePropagate.${MEXSUFFIX}: ${SRC}/cuCollidePropagate.cu \
		${SRC}/barracuda.cu ${SRC}/barracuda.h
	@echo ---[ ${SRC}/cuCollidePropagate.cu ]------ ${LOG_ECHO}
	@$(NVMEX) -f $(NVOPTS) ${INC_CUDA} \
	 ${LIB_CUDA} ${LIBS_CUDA} \
	${SRC}/cuCollidePropagate.cu ${SRC}/barracuda.cu \
	-o ${BIN}/cuCollidePropagate ${LOG_ERR}

#---------------------------------------------------[cuDevicesProperties.cu]---
${BIN}/cuDeviceProperties.${MEXSUFFIX}: ${SRC}/cuDeviceProperties.cu
	@echo ---[ ${SRC}/cuDeviceProperties.cu ]------ ${LOG_ECHO}
	@$(NVMEX) -f $(NVOPTS) ${INC_CUDA} \
	 ${LIB_CUDA} ${LIBS_CUDA} \
	${SRC}/cuDeviceProperties.cu \
	-o ${BIN}/cuDeviceProperties ${LOG_ERR}

#--------------------------------------------------------[cuDevicesReset.cu]---
${BIN}/cuDeviceReset.${MEXSUFFIX}: ${SRC}/cuDeviceReset.cu
	@echo ---[ ${SRC}/cuDeviceReset.cu ]------ ${LOG_ECHO}
	@$(NVMEX) -f $(NVOPTS) ${INC_CUDA} \
	 ${LIB_CUDA} ${LIBS_CUDA} \
	${SRC}/cuDeviceReset.cu \
	-o ${BIN}/cuDeviceReset ${LOG_ERR}

#----------------------------------------------------------------[init-make]---
.PHONY: init-make
init-make:
	@rm -f make.log

#--------------------------------------------------------------------[clean]---
.PHONY: clean
clean:
	@rm -f ${BIN}/*.${MEXSUFFIX}
	@rm -f make.log

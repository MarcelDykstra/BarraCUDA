#include <stdio.h>
#include "barracuda.h"

//=============================================================================
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{

  // check for proper number of arguments
  if (nrhs != 0) {
    mexErrMsgTxt("Too many input arguments.");
  } else if (nlhs > 0) {
    mexErrMsgTxt("Too many output arguments.");
  }

  cudaDeviceReset();
}


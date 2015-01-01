#include <stdio.h>
#include "barracuda.h"

//==============================================================================
void cuLatticeClear(void)
{
  unsigned int n;

  cudaFree(state.dev_map);

  cudaFree(state.dev_vx);
  cudaFree(state.dev_vy);
  cudaFree(state.dev_rho);

  for (n = 0; n < N_FLUID; n++) {
    cudaFree(state.dev_dist1[n].fC);
    cudaFree(state.dev_dist1[n].fE);
    cudaFree(state.dev_dist1[n].fW);
    cudaFree(state.dev_dist1[n].fS);
    cudaFree(state.dev_dist1[n].fN);
    cudaFree(state.dev_dist1[n].fNE);
    cudaFree(state.dev_dist1[n].fNW);
    cudaFree(state.dev_dist1[n].fSE);
    cudaFree(state.dev_dist1[n].fSW);

    cudaFree(state.dev_dist2[n].fC);
    cudaFree(state.dev_dist2[n].fE);
    cudaFree(state.dev_dist2[n].fW);
    cudaFree(state.dev_dist2[n].fS);
    cudaFree(state.dev_dist2[n].fN);
    cudaFree(state.dev_dist2[n].fNE);
    cudaFree(state.dev_dist2[n].fNW);
    cudaFree(state.dev_dist2[n].fSE);
    cudaFree(state.dev_dist2[n].fSW);
  }

}

//==============================================================================
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // Check for proper number of arguments
  if (nrhs != 1) {
    mexErrMsgTxt("One input required.");
  } else if (nlhs > 0) {
    mexErrMsgTxt("Too many output arguments.");
  }

  if (!readState(prhs)) mexErrMsgTxt("Invalid CUDA handle.");

  cuLatticeClear();

  destroyState();
}


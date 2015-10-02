#include <stdio.h>
#include "barracuda.h"

state_t state;
state_t *dev_state;

//==============================================================================
void createState(void)
{
  cudaMalloc((void **) &dev_state, sizeof(state_t));
}

//==============================================================================
void destroyState(void)
{
  cudaFree(dev_state);
}

//==============================================================================
void writeState(mxArray *plhs[])
{
  long long *ptr;

  state.magic = 0xBAACDA;

  cudaMemcpy(dev_state, &state, sizeof(state_t), cudaMemcpyHostToDevice);

  ptr = (long long *) mxMalloc(sizeof(long));
  ptr[0] = (long long) dev_state;
  plhs[0] = mxCreateNumericArray(0, 0, mxUINT64_CLASS, mxREAL);
  mxSetPr(plhs[0], (double *) ptr);
  mxSetM(plhs[0], 1); mxSetN(plhs[0], 1);
}

//==============================================================================
void syncState(void)
{
  cudaMemcpy(dev_state, &state, sizeof(state_t), cudaMemcpyHostToDevice);
}

//==============================================================================
bool readState(const mxArray *prhs[])
{
  long long *ptr;

  ptr = (long long *) mxGetData(prhs[0]);
  dev_state = (state_t *) ptr[0];

  cudaMemcpy(&state, dev_state, sizeof(state_t), cudaMemcpyDeviceToHost);

  if (state.magic != 0xBAACDA) return false;
  return true;
}

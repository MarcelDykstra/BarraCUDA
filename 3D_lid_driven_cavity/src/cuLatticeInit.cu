#include <stdio.h>
#include "barracuda.h"

//=============================================================================
void cuLatticeInit(void)
{
  float *lat[3];
  unsigned int *map;
  unsigned int n, w, d, h;

  cudaMalloc((void **) &(state.dev_map), SIZE_I);

  cudaMalloc((void **) &(state.dev_vx), SIZE_F);
  cudaMalloc((void **) &(state.dev_vy), SIZE_F);
  cudaMalloc((void **) &(state.dev_vz), SIZE_F);
  cudaMalloc((void **) &(state.dev_rho), SIZE_F);

  for (n = 0; n < N_FLUID; n++) {
    cudaMalloc((void **) &(state.dev_dist1[n].fC), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fE), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fW), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fN), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fS), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fNE), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fSE), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fNW), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fSW), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fU),  SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fUE), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fUW), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fUN), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fUS), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fD),  SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fDE), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fDW), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fDN), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist1[n].fDS), SIZE_F);

    cudaMalloc((void **) &(state.dev_dist2[n].fC), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fE), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fW), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fN), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fS), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fNE), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fSE), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fNW), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fSW), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fU),  SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fUE), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fUW), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fUN), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fUS), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fD),  SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fDE), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fDW), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fDN), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fDS), SIZE_F);
  }

  for (n = 0; n < 3; n++) lat[n] = (float *) mxMalloc(SIZE_F);

  for (w = 0; w < LAT_SIZE_W; w++) {
    for (d = 0; d < LAT_SIZE_D; d++) {
      for (h = 0; h < LAT_SIZE_H; h++) {
        lat[0][GBL_IDX(w, d, h)] = 12.0f / 36.0f;
        lat[1][GBL_IDX(w, d, h)] = 2.0f / 36.0f;
        lat[2][GBL_IDX(w, d, h)] = 1.0f / 36.0f;
      }
    }
  }

  map = (unsigned int*) mxMalloc(SIZE_I);

  // fluid
  for (w = 0; w < LAT_SIZE_W; w++) {
    for (d = 0; d <LAT_SIZE_D; d++) {
      for (h = 0; h < LAT_SIZE_H; h++) {
        map[GBL_IDX(w, d, h)] = GEO_FLUID;
      }
    }
  }

  // top/bottom
  for(w = 0; w < LAT_SIZE_W; w++) {
    for(d = 0; d < LAT_SIZE_D; d++) {
      map[GBL_IDX(w, d, 0)] = map[GBL_IDX(w, d, LAT_SIZE_H - 1)] = GEO_WALL;
    }
  }

  // front/back
  for(w = 0; w < LAT_SIZE_W; w++) {
    for(h = 0; h < LAT_SIZE_H; h++) {
      map[GBL_IDX(w, 0, h)] = GEO_INFLOW;
      map[GBL_IDX(w, LAT_SIZE_D - 1, h)] = GEO_WALL;
    }
  }

  // left/right
  for(d = 0;d < LAT_SIZE_D; d++) {
    for(h = 0; h < LAT_SIZE_H; h++) {
      map[GBL_IDX(0, d, h)] = map[GBL_IDX(LAT_SIZE_W - 1, d, h)] = GEO_WALL;
    }
  }

  cudaMemcpy(state.dev_map, map, SIZE_I, cudaMemcpyHostToDevice);

  for(n = 0;n < N_FLUID; n++) {
    cudaMemcpy(state.dev_dist1[n].fC, lat[0], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fN, lat[1], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fS, lat[1], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fE, lat[1], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fW, lat[1], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fNE, lat[2], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fNW, lat[2], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fSE, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fSW, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fU, lat[1] ,SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fUE, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fUW, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fUN, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fUS, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fD, lat[1] ,SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fDE, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fDW, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fDN, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fDS, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);

    cudaMemcpy(state.dev_dist2[n].fC, lat[0],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fN, lat[1],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fS, lat[1],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fE, lat[1],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fW, lat[1],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fNE, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fNW, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fSE, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fSW, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fU, lat[1] ,SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fUE, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fUW, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fUN, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fUS, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fD, lat[1] ,SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fDE, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fDW, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fDN, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fDS, lat[2],SIZE_F,
               cudaMemcpyHostToDevice);
  }

  for (n = 0;n < 3; n++) mxFree(lat[n]);
  mxFree(map);
}

//=============================================================================
void cudaDeviceProperties(cudaDeviceProp *prop) {
  int count;
  cudaGetDeviceCount(&count);
  for (int i = 0; i < count; i++) {
    cudaGetDeviceProperties(prop, i);
  }
}

//=============================================================================
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  cudaDeviceProp prop;

  // check for proper number of arguments
  if (nrhs != 0) {
    mexErrMsgTxt("Too many input arguments.");
  } else if (nlhs > 1) {
    mexErrMsgTxt("Too many output arguments.");
  }

  cudaDeviceProperties(&prop);

  // check agains hardware
  if (BLOCK_SIZE_W * BLOCK_SIZE_D * BLOCK_SIZE_H > prop.maxThreadsPerBlock)
    mexErrMsgTxt("Too many threads per block.");

  createState();

  cuLatticeInit();

  writeState(plhs);
}

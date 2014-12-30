#include <stdio.h>
#include "barracuda.h"

//=============================================================================
void cuLatticeInit(void)
{
  float *lat[9];
  unsigned int *map;
  unsigned int n, m;

  cudaMalloc((void **) &(state.dev_map), SIZE_I);

  cudaMalloc((void **) &(state.dev_vx), SIZE_F);
  cudaMalloc((void **) &(state.dev_vy), SIZE_F);
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

    cudaMalloc((void **) &(state.dev_dist2[n].fC), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fE), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fW), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fN), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fS), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fNE), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fSE), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fNW), SIZE_F);
    cudaMalloc((void **) &(state.dev_dist2[n].fSW), SIZE_F);
  }

  for (n = 0; n < 9; n++) lat[n] = (float *) mxMalloc(SIZE_F);

  for (n = 0; n < LAT_W; n++) {
   for (m = 0; m < LAT_H; m++) {
    lat[0][n + m * LAT_W] = 4.0 / 9.0 ;
    lat[1][n + m * LAT_W] = lat[2][n + m * LAT_W] =
    lat[3][n + m * LAT_W] = lat[4][n + m * LAT_W] = 1.0 / 9.0;
    lat[5][n + m * LAT_W] = lat[6][n + m * LAT_W] =
    lat[7][n + m * LAT_W] = lat[8][n + m * LAT_W] = 1.0 / 36.0;
   }
  }

  map = (unsigned int*) mxMalloc(SIZE_I);

  // fluid
  for (n = 0; n < LAT_W; n++) {
    for (m = 0; m < LAT_H; m++) {
      map[n + m * LAT_W] = GEO_FLUID;
    }
  }

  // top/bottom
  for (n = 0; n < LAT_W; n++) {
    map[n] = GEO_WALL;
  }

  // left /right
  for (n = 0; n < LAT_H; n++) {
    map[n * LAT_W] = map[LAT_W - 1 + n * LAT_W] = GEO_WALL;
  }

  //top
  for (n = 0; n < LAT_W; n++) {
    map[(LAT_H - 1) * LAT_W + n] = GEO_INFLOW;
  }

  cudaMemcpy(state.dev_map,map, SIZE_I, cudaMemcpyHostToDevice);

  for (n = 0; n < N_FLUID; n++) {
    cudaMemcpy(state.dev_dist1[n].fC, lat[0], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fN, lat[1], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fS, lat[2], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fE, lat[3], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fW, lat[4], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fNE, lat[5], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fNW, lat[6], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fSE, lat[7], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist1[n].fSW, lat[8], SIZE_F,
               cudaMemcpyHostToDevice);

    cudaMemcpy(state.dev_dist2[n].fC, lat[0], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fN, lat[1], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fS, lat[2], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fE, lat[3], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fW, lat[4], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fNE, lat[5], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fNW, lat[6], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fSE, lat[7], SIZE_F,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.dev_dist2[n].fSW, lat[8], SIZE_F,
               cudaMemcpyHostToDevice);
  }

  for (n = 0; n < 9; n++) mxFree(lat[n]);
  mxFree(map);
}

//=============================================================================
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // check for proper number of arguments
  if (nrhs != 0) {
    mexErrMsgTxt("Too many input arguments.");
  } else if (nlhs > 1) {
    mexErrMsgTxt("Too many output arguments.");
  }

  createState();

  cuLatticeInit();

  writeState(plhs);
}

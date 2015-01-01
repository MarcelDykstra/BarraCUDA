#include <stdio.h>
#include "barracuda.h"

//==============================================================================
__global__ void knCollidePropogate(unsigned int *map, dist_ptr_t dist_in[],
                                   dist_ptr_t dist_out[], float *rho_out,
                                   float *vx_out, float *vy_out, float *vz_out)
{
  int wix = threadIdx.x + blockIdx.x * blockDim.x;
  int dix = threadIdx.y + blockIdx.y * blockDim.y;
  int hix = threadIdx.z + blockIdx.z * blockDim.z;
  int idx = GBL_IDX(wix, dix, hix);
  int n;

  // Equilibrium distributions
  dist_t dist_eq[N_FLUID];
  dist_t dist_thd[N_FLUID];

  float  rho[N_FLUID];
  float3 v[N_FLUID];

  const float tau[N_FLUID] = {TAU_1, TAU_2};

  // Shared variables for in-block propagation
  ///__shared__ propa_t dist_blk[BLOCK_SIZE_W * BLOCK_SIZE_D * BLOCK_SIZE_H];

  // Cache the distribution in local variables
  for (n = 0; n < N_FLUID; n++) {
    dist_thd[n].fC  = dist_in[n].fC [idx];
    dist_thd[n].fE  = dist_in[n].fE [idx];
    dist_thd[n].fW  = dist_in[n].fW [idx];
    dist_thd[n].fS  = dist_in[n].fS [idx];
    dist_thd[n].fN  = dist_in[n].fN [idx];
    dist_thd[n].fNE = dist_in[n].fNE[idx];
    dist_thd[n].fNW = dist_in[n].fNW[idx];
    dist_thd[n].fSE = dist_in[n].fSE[idx];
    dist_thd[n].fSW = dist_in[n].fSW[idx];
    dist_thd[n].fU  = dist_in[n].fU [idx];
    dist_thd[n].fUE = dist_in[n].fUE[idx];
    dist_thd[n].fUW = dist_in[n].fUW[idx];
    dist_thd[n].fUN = dist_in[n].fUN[idx];
    dist_thd[n].fUS = dist_in[n].fUS[idx];
    dist_thd[n].fD  = dist_in[n].fD [idx];
    dist_thd[n].fDE = dist_in[n].fDE[idx];
    dist_thd[n].fDW = dist_in[n].fDW[idx];
    dist_thd[n].fDN = dist_in[n].fDN[idx];
    dist_thd[n].fDS = dist_in[n].fDS[idx];
  }

  // Macroscopic quantities for the current cell
  for (n = 0; n < N_FLUID; n++) {
    rho[n] = dist_thd[n].fC  + dist_thd[n].fE  + dist_thd[n].fW  +
             dist_thd[n].fS  + dist_thd[n].fN  + dist_thd[n].fNE +
             dist_thd[n].fNW + dist_thd[n].fSE + dist_thd[n].fSW +
             dist_thd[n].fU  + dist_thd[n].fUE + dist_thd[n].fUW +
             dist_thd[n].fUN + dist_thd[n].fUS + dist_thd[n].fD  +
             dist_thd[n].fDE + dist_thd[n].fDW + dist_thd[n].fDN +
             dist_thd[n].fDS ;

    if (map[idx] == GEO_INFLOW) {
      v[n].x = 0.1f;
      v[n].y = 0.0f;
      v[n].z = 0.0f;
    }
    else {
      v[n].x = (dist_thd[n].fE  - dist_thd[n].fW  + dist_thd[n].fSE -
                dist_thd[n].fSW + dist_thd[n].fNE - dist_thd[n].fNW +
                dist_thd[n].fUE - dist_thd[n].fUW + dist_thd[n].fDE -
                dist_thd[n].fDW) / rho[n];
      v[n].y = (dist_thd[n].fN  - dist_thd[n].fS  + dist_thd[n].fNE -
                dist_thd[n].fSE + dist_thd[n].fNW - dist_thd[n].fSW +
                dist_thd[n].fUN - dist_thd[n].fUS + dist_thd[n].fDN -
                dist_thd[n].fDS) / rho[n];
      v[n].z = (dist_thd[n].fU  - dist_thd[n].fD  + dist_thd[n].fUS -
                dist_thd[n].fDS + dist_thd[n].fUN - dist_thd[n].fDN +
                dist_thd[n].fUE - dist_thd[n].fDE + dist_thd[n].fUW -
                dist_thd[n].fDW) / rho[n];
    }
  }

  if (rho_out != NULL) {
    rho_out[idx] = rho[0];
    vx_out[idx]  = v[0].x;
    vy_out[idx]  = v[1].y;
    vz_out[idx]  = v[0].z;
  }

  // Relaxation
  float Vsq, VCsq;
  const float Csq = 1.0f / 3.0f; // Lattice "speed of sound"
  const float Cdsq = Csq * Csq;

  for (n = 0; n < N_FLUID; n++) {
    Vsq = (v[n].x * v[n].x + v[n].y * v[n].y + v[n].z * v[n].z);
    VCsq = Vsq / (2.0f * Csq);

    dist_eq[n].fC  = 12.0f / 36.0f * rho[n] * (1.0f - VCsq);
    dist_eq[n].fS  = 2.0f / 36.0f  * rho[n] * (1.0f - v[n].y / Csq +
                       (v[n].y * v[n].y) / (2.0f * Cdsq) - VCsq);
    dist_eq[n].fN  = 2.0f / 36.0f  * rho[n] * (1.0f + v[n].y / Csq +
                       (v[n].y * v[n].y) / (2.0f * Cdsq) - VCsq);
    dist_eq[n].fE  = 2.0f / 36.0f  * rho[n] * (1.0f + v[n].x / Csq +
                       (v[n].x * v[n].x) / (2.0f * Cdsq) - VCsq);
    dist_eq[n].fW  = 2.0f / 36.0f  * rho[n] * (1.0f - v[n].x / Csq +
                       (v[n].x * v[n].x) / (2.0f * Cdsq) - VCsq);
    dist_eq[n].fU  = 2.0f / 36.0f  * rho[n] * (1.0f + v[n].z / Csq +
                       (v[n].z * v[n].z) / (2.0f * Cdsq) - VCsq);
    dist_eq[n].fD  = 2.0f / 36.0f  * rho[n] * (1.0f - v[n].z / Csq +
                       (v[n].z * v[n].z) / (2.0f * Cdsq) - VCsq);
    dist_eq[n].fSE = 1.0f / 36.0f  * rho[n] *
                       (1.0f + (-v[n].y + v[n].x) / Csq +
                       (-v[n].y + v[n].x) * (-v[n].y + v[n].x) /
                       (2.0f * Cdsq) - VCsq);
    dist_eq[n].fSW = 1.0f / 36.0f * rho[n] *
                       (1.0f + (-v[n].y - v[n].x) / Csq +
                       (-v[n].y - v[n].x) * (-v[n].y - v[n].x) /
                       (2.0f * Cdsq) - VCsq);
    dist_eq[n].fUS = 1.0f / 36.0f  * rho[n] *
                       (1.0f + (-v[n].y + v[n].z) / Csq +
                       (-v[n].y + v[n].z) * (-v[n].y + v[n].z) /
                       (2.0f * Cdsq) - VCsq);
    dist_eq[n].fDS = 1.0f / 36.0f  * rho[n] *
                       (1.0f + (-v[n].y - v[n].z) / Csq +
                       (-v[n].y - v[n].z) * (-v[n].y - v[n].z) /
                       (2.0f * Cdsq) - VCsq);
    dist_eq[n].fNE = 1.0f / 36.0f  * rho[n] *
                       (1.0f + (v[n].y + v[n].x) / Csq +
                       (v[n].y + v[n].x) * (v[n].y + v[n].x) /
                       (2.0f * Cdsq) - VCsq);
    dist_eq[n].fNW = 1.0f / 36.0f  * rho[n] *
                       (1.0f + (v[n].y - v[n].x) / Csq +
                       (v[n].y - v[n].x) * (v[n].y - v[n].x) /
                       (2.0f * Cdsq) - VCsq);
    dist_eq[n].fUN = 1.0f / 36.0f  * rho[n] *
                       (1.0f + (v[n].y + v[n].z) / Csq +
                       (v[n].y + v[n].z) * (v[n].y + v[n].z) /
                       (2.0f * Cdsq) - VCsq);
    dist_eq[n].fDN = 1.0f / 36.0f  * rho[n] *
                       (1.0f + (v[n].y - v[n].z) / Csq +
                       (v[n].y - v[n].z) * (v[n].y - v[n].z) /
                       (2.0f * Cdsq) - VCsq);
    dist_eq[n].fUE = 1.0f / 36.0f  * rho[n] *
                       (1.0f + (v[n].x + v[n].z) / Csq +
                       (v[n].x + v[n].z) * (v[n].x + v[n].z) /
                       (2.0f * Cdsq) - VCsq);
    dist_eq[n].fDE = 1.0f / 36.0f  * rho[n] *
                       (1.0f + (v[n].x - v[n].z) / Csq +
                       (v[n].x - v[n].z) * (v[n].x - v[n].z) /
                       (2.0f * Cdsq) - VCsq);
    dist_eq[n].fUW = 1.0f / 36.0f  * rho[n] *
                       (1.0f + (-v[n].x + v[n].z) / Csq +
                       (-v[n].x + v[n].z) * (-v[n].x + v[n].z) /
                       (2.0f*Cdsq) - VCsq);
    dist_eq[n].fDW = 1.0f / 36.0f  * rho[n] *
                       (1.0f + (-v[n].x - v[n].z) / Csq +
                       (-v[n].x - v[n].z) * (-v[n].x - v[n].z) /
                       (2.0f * Cdsq) - VCsq);

    if (map[idx] == GEO_FLUID) {
      dist_thd[n].fC  += (dist_eq[n].fC  - dist_thd[n].fC)  / tau[n];
      dist_thd[n].fE  += (dist_eq[n].fE  - dist_thd[n].fE)  / tau[n];
      dist_thd[n].fW  += (dist_eq[n].fW  - dist_thd[n].fW)  / tau[n];
      dist_thd[n].fS  += (dist_eq[n].fS  - dist_thd[n].fS)  / tau[n];
      dist_thd[n].fN  += (dist_eq[n].fN  - dist_thd[n].fN)  / tau[n];
      dist_thd[n].fSE += (dist_eq[n].fSE - dist_thd[n].fSE) / tau[n];
      dist_thd[n].fNE += (dist_eq[n].fNE - dist_thd[n].fNE) / tau[n];
      dist_thd[n].fSW += (dist_eq[n].fSW - dist_thd[n].fSW) / tau[n];
      dist_thd[n].fNW += (dist_eq[n].fNW - dist_thd[n].fNW) / tau[n];
      dist_thd[n].fU  += (dist_eq[n].fU  - dist_thd[n].fU)  / tau[n];
      dist_thd[n].fUE += (dist_eq[n].fUE - dist_thd[n].fUE) / tau[n];
      dist_thd[n].fUW += (dist_eq[n].fUW - dist_thd[n].fUW) / tau[n];
      dist_thd[n].fUN += (dist_eq[n].fUN - dist_thd[n].fUN) / tau[n];
      dist_thd[n].fUS += (dist_eq[n].fUS - dist_thd[n].fUS) / tau[n];
      dist_thd[n].fD  += (dist_eq[n].fD  - dist_thd[n].fD)  / tau[n];
      dist_thd[n].fDE += (dist_eq[n].fDE - dist_thd[n].fDE) / tau[n];
      dist_thd[n].fDW += (dist_eq[n].fDW - dist_thd[n].fDW) / tau[n];
      dist_thd[n].fDN += (dist_eq[n].fDN - dist_thd[n].fDN) / tau[n];
      dist_thd[n].fDS += (dist_eq[n].fDS - dist_thd[n].fDS) / tau[n];
    }
    else if (map[idx] == GEO_INFLOW) {
      dist_thd[n].fC  = dist_eq[n].fC;
      dist_thd[n].fE  = dist_eq[n].fE;
      dist_thd[n].fW  = dist_eq[n].fW;
      dist_thd[n].fS  = dist_eq[n].fS;
      dist_thd[n].fN  = dist_eq[n].fN;
      dist_thd[n].fSE = dist_eq[n].fSE;
      dist_thd[n].fNE = dist_eq[n].fNE;
      dist_thd[n].fSW = dist_eq[n].fSW;
      dist_thd[n].fNW = dist_eq[n].fNW;
      dist_thd[n].fU  = dist_eq[n].fU;
      dist_thd[n].fUE = dist_eq[n].fUE;
      dist_thd[n].fUW = dist_eq[n].fUW;
      dist_thd[n].fUN = dist_eq[n].fUN;
      dist_thd[n].fUS = dist_eq[n].fUS;
      dist_thd[n].fD  = dist_eq[n].fD;
      dist_thd[n].fDE = dist_eq[n].fDE;
      dist_thd[n].fDW = dist_eq[n].fDW;
      dist_thd[n].fDN = dist_eq[n].fDN;
      dist_thd[n].fDS = dist_eq[n].fDS;
    }
    else if (map[idx] == GEO_WALL) {
      float swp;
      swp = dist_thd[n].fE;
      dist_thd[n].fE = dist_thd[n].fW;
      dist_thd[n].fW = swp;

      swp = dist_thd[n].fNW;
      dist_thd[n].fNW = dist_thd[n].fSE;
      dist_thd[n].fSE = swp;

      swp = dist_thd[n].fNE;
      dist_thd[n].fNE = dist_thd[n].fSW;
      dist_thd[n].fSW = swp;

      swp = dist_thd[n].fN;
      dist_thd[n].fN = dist_thd[n].fS;
      dist_thd[n].fS = swp;

      swp = dist_thd[n].fU;
      dist_thd[n].fU = dist_thd[n].fD;
      dist_thd[n].fD = swp;

      swp = dist_thd[n].fUE;
      dist_thd[n].fUE = dist_thd[n].fDW;
      dist_thd[n].fDW = swp;

      swp = dist_thd[n].fUW;
      dist_thd[n].fUW = dist_thd[n].fDE;
      dist_thd[n].fDE = swp;

      swp = dist_thd[n].fUN;
      dist_thd[n].fUN = dist_thd[n].fDS;
      dist_thd[n].fDS = swp;

      swp = dist_thd[n].fUS;
      dist_thd[n].fUS = dist_thd[n].fDN;
      dist_thd[n].fDN = swp;
    }
  }

  for (n = 0; n < N_FLUID; n++) {
    dist_out[n].fC[idx] = dist_thd[n].fC;
    dist_out[n].fE [GBL_IDX(MOD(wix + 1, LAT_SIZE_W), dix, hix)]
      = dist_thd[n].fE;
    dist_out[n].fW [GBL_IDX(MOD(wix - 1, LAT_SIZE_W), dix, hix)]
      = dist_thd[n].fW;
    dist_out[n].fS [GBL_IDX(wix, MOD(dix - 1, LAT_SIZE_D), hix)]
      = dist_thd[n].fS;
    dist_out[n].fN [GBL_IDX(wix, MOD(dix + 1, LAT_SIZE_D), hix)]
      = dist_thd[n].fN;
    dist_out[n].fSE[
      GBL_IDX(MOD(wix + 1, LAT_SIZE_W), MOD(dix - 1, LAT_SIZE_D), hix)]
      = dist_thd[n].fSE;
    dist_out[n].fNE[
      GBL_IDX(MOD(wix + 1, LAT_SIZE_W), MOD(dix + 1, LAT_SIZE_D), hix)]
      = dist_thd[n].fNE;
    dist_out[n].fSW[
      GBL_IDX(MOD(wix - 1, LAT_SIZE_W), MOD(dix - 1, LAT_SIZE_D), hix)]
      = dist_thd[n].fSW;
    dist_out[n].fNW[
      GBL_IDX(MOD(wix - 1, LAT_SIZE_W), MOD(dix + 1, LAT_SIZE_D), hix)]
      = dist_thd[n].fNW;
    dist_out[n].fU [GBL_IDX(wix, dix, MOD(hix + 1, LAT_SIZE_H))]
      = dist_thd[n].fU;
    dist_out[n].fUE[
      GBL_IDX(MOD(wix + 1, LAT_SIZE_W), dix, MOD(hix + 1, LAT_SIZE_H))]
      = dist_thd[n].fUE;
    dist_out[n].fUW[
      GBL_IDX(MOD(wix - 1, LAT_SIZE_W), dix,MOD(hix + 1, LAT_SIZE_H))]
      = dist_thd[n].fUW;
    dist_out[n].fUN[
      GBL_IDX(wix, MOD(dix + 1, LAT_SIZE_D), MOD(hix + 1, LAT_SIZE_H))]
      = dist_thd[n].fUN;
    dist_out[n].fUS[
      GBL_IDX(wix, MOD(dix - 1, LAT_SIZE_D), MOD(hix + 1, LAT_SIZE_H))]
      = dist_thd[n].fUS;
    dist_out[n].fD [GBL_IDX(wix, dix, MOD(hix - 1, LAT_SIZE_H))]
      = dist_thd[n].fD;
    dist_out[n].fDE[
      GBL_IDX(MOD(wix + 1, LAT_SIZE_W), dix, MOD(hix - 1, LAT_SIZE_H))]
      = dist_thd[n].fDE;
    dist_out[n].fDW[
      GBL_IDX(MOD(wix - 1, LAT_SIZE_W), dix, MOD(hix - 1, LAT_SIZE_H))]
      = dist_thd[n].fDW;
    dist_out[n].fDN[
      GBL_IDX(wix, MOD(dix + 1, LAT_SIZE_D), MOD(hix - 1, LAT_SIZE_H))]
      = dist_thd[n].fDN;
    dist_out[n].fDS[
      GBL_IDX(wix, MOD(dix - 1, LAT_SIZE_D), MOD(hix - 1, LAT_SIZE_H))]
      = dist_thd[n].fDS;
  }
}

//==============================================================================
void cuCollidePropagate(void)
{
  unsigned int n;
  dim3 grid, block;

  grid.x = GRID_SIZE_W;
  grid.y = GRID_SIZE_D;
  grid.z = GRID_SIZE_H;

  block.x = BLOCK_SIZE_W;
  block.y = BLOCK_SIZE_D;
  block.z = BLOCK_SIZE_H;

  for (n = 0; n < 100; n++) {
    if (n == 99) {
      knCollidePropogate <<<grid,block>>>
        (state.dev_map, dev_state->dev_dist2, dev_state->dev_dist1,
         state.dev_rho, state.dev_vx, state.dev_vy, state.dev_vz);
    }
    else if (n % 2 == 0) {
      knCollidePropogate <<<grid,block>>>
        (state.dev_map, dev_state->dev_dist1, dev_state->dev_dist2,
         NULL, NULL, NULL, NULL);
    }
    else {
      knCollidePropogate <<<grid,block>>>
        (state.dev_map, dev_state->dev_dist2, dev_state->dev_dist1,
         NULL, NULL, NULL, NULL);
    }
  }

}

//==============================================================================
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  float *param;
  mwSize dims[3] = {LAT_SIZE_W, LAT_SIZE_D, LAT_SIZE_H};

  // Check for proper number of arguments
  if (nrhs != 1) {
    mexErrMsgTxt("One input required.");
  } else if (nlhs > 4) {
    mexErrMsgTxt("Too many output arguments.");
  }

  if (!readState(prhs)) mexErrMsgTxt("Invalid CUDA handle.");

  cuCollidePropagate();

  if (nlhs >= 1) {
   plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
   param = (float*) mxGetData(plhs[0]);
   cudaMemcpy(param, state.dev_vx, SIZE_F, cudaMemcpyDeviceToHost);
  }

  if (nlhs >= 2) {
   plhs[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
   param = (float*) mxGetData(plhs[1]);
   cudaMemcpy(param, state.dev_vy, SIZE_F, cudaMemcpyDeviceToHost);
  }


  if (nlhs >= 3) {
   plhs[2] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
   param = (float*) mxGetData(plhs[2]);
   cudaMemcpy(param, state.dev_vz, SIZE_F, cudaMemcpyDeviceToHost);
  }


  if (nlhs >= 4) {
   plhs[3] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
   param = (float*) mxGetData(plhs[3]);
   cudaMemcpy(param, state.dev_rho, SIZE_F, cudaMemcpyDeviceToHost);
  }

  syncState();
}

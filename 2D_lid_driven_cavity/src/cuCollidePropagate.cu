#include <stdio.h>
#include "barracuda.h"

//------------------------------------------------------------------------------
__global__ void knCollidePropogate(unsigned int *map, dist_t cd[], dist_t od[],
                                   float *orho, float *ovx, float *ovy)
{
  int tix = threadIdx.x;
  int ti = tix + blockIdx.x * blockDim.x;
  int gi = ti + LAT_W * blockIdx.y;
  int n;

  // Equilibrium distributions
  float feq_C[N_FLUID], feq_N[N_FLUID], feq_S[N_FLUID], feq_E[N_FLUID];
  float feq_W[N_FLUID], feq_NE[N_FLUID], feq_NW[N_FLUID], feq_SE[N_FLUID];
  float feq_SW[N_FLUID];
  float fi_N[N_FLUID], fi_S[N_FLUID], fi_E[N_FLUID], fi_W[N_FLUID];
  float fi_C[N_FLUID], fi_NE[N_FLUID], fi_NW[N_FLUID], fi_SE[N_FLUID];
  float fi_SW[N_FLUID];
  float rho[N_FLUID];
  float2 v[N_FLUID];

  const float tau[N_FLUID] = {TAU_1}; ///, TAU_2};

  // Shared variables for in-block propagation
  __shared__ float fo_E[BLOCK_SIZE];
  __shared__ float fo_W[BLOCK_SIZE];
  __shared__ float fo_SE[BLOCK_SIZE];
  __shared__ float fo_SW[BLOCK_SIZE];
  __shared__ float fo_NE[BLOCK_SIZE];
  __shared__ float fo_NW[BLOCK_SIZE];

  // Cache the distribution in local variables
  for (n = 0; n < N_FLUID; n++) {
    fi_C[n] = cd[n].fC[gi];
    fi_E[n] = cd[n].fE[gi];
    fi_W[n] = cd[n].fW[gi];
    fi_S[n] = cd[n].fS[gi];
    fi_N[n] = cd[n].fN[gi];
    fi_NE[n] = cd[n].fNE[gi];
    fi_NW[n] = cd[n].fNW[gi];
    fi_SE[n] = cd[n].fSE[gi];
    fi_SW[n] = cd[n].fSW[gi];
  }

  // Macroscopic quantities for the current cell
  for (n = 0; n < N_FLUID; n++) {
    rho[n] = fi_C[n] + fi_E[n] + fi_W[n] + fi_S[n] + fi_N[n] + fi_NE[n] +
             fi_NW[n] + fi_SE[n] + fi_SW[n];
    if (map[gi] == GEO_INFLOW) {
      v[n].x = 0.1f;
      v[n].y = 0.0f;
    }
    else {
      v[n].x = (fi_E[n] + fi_SE[n] + fi_NE[n] - fi_W[n] -
                fi_SW[n] - fi_NW[n]) / rho[n];
      v[n].y = (fi_N[n] + fi_NW[n] + fi_NE[n] - fi_S[n] -
                fi_SW[n] - fi_SE[n]) / rho[n];
    }
  }

  if (orho != NULL) {
    orho[gi] = rho[0];
    ovx[gi] = v[0].x;
    ovy[gi] = v[0].y;
  }

  // Relaxation
  float Cusq[N_FLUID];

  for (n = 0; n < N_FLUID; n++) {
    Cusq[n] = -1.5f * (v[n].x*v[n].x + v[n].y*v[n].y);

    feq_C[n] = rho[n] * (1.0f + Cusq[n]) * 4.0f / 9.0f;
    feq_N[n] = rho[n] * (1.0f + Cusq[n] + 3.0f *  v[n].y +
                         4.5f * v[n].y * v[n].y) / 9.0f;
    feq_E[n] = rho[n] * (1.0f + Cusq[n] + 3.0f * v[n].x +
                         4.5f * v[n].x*v[n].x) / 9.0f;
    feq_S[n] = rho[n] * (1.0f + Cusq[n] - 3.0f * v[n].y +
                         4.5f * v[n].y * v[n].y) / 9.0f;
    feq_W[n] = rho[n] * (1.0f + Cusq[n] - 3.0f*v[n].x +
                         4.5f*v[n].x*v[n].x) / 9.0f;
    feq_NE[n] = rho[n] * (1.0f + Cusq[n] + 3.0f * (v[n].x + v[n].y) +
                          4.5f * (v[n].x + v[n].y) * (v[n].x + v[n].y)) /
                          36.0f;
    feq_SE[n] = rho[n] * (1.0f + Cusq[n] + 3.0f * (v[n].x - v[n].y) +
                          4.5f * (v[n].x - v[n].y) * (v[n].x - v[n].y)) /
                          36.0f;
    feq_SW[n] = rho[n] * (1.0f + Cusq[n] + 3.0f * (-v[n].x - v[n].y) +
                          4.5f * (v[n].x + v[n].y) * (v[n].x + v[n].y)) /
                          36.0f;
    feq_NW[n] = rho[n] * (1.0f + Cusq[n] + 3.0f * (-v[n].x + v[n].y) +
                          4.5f * (-v[n].x + v[n].y) * (-v[n].x + v[n].y)) /
                          36.0f;

    if (map[gi] == GEO_FLUID) {
      fi_C[n] += (feq_C[n] - fi_C[n]) / tau[n];
      fi_E[n] += (feq_E[n] - fi_E[n]) / tau[n];
      fi_W[n] += (feq_W[n] - fi_W[n]) / tau[n];
      fi_S[n] += (feq_S[n] - fi_S[n]) / tau[n];
      fi_N[n] += (feq_N[n] - fi_N[n]) / tau[n];
      fi_SE[n] += (feq_SE[n] - fi_SE[n]) / tau[n];
      fi_NE[n] += (feq_NE[n] - fi_NE[n]) / tau[n];
      fi_SW[n] += (feq_SW[n] - fi_SW[n]) / tau[n];
      fi_NW[n] += (feq_NW[n] - fi_NW[n]) / tau[n];
    }
    else if (map[gi] == GEO_INFLOW) {
      fi_C[n]  = feq_C[n];
      fi_E[n]  = feq_E[n];
      fi_W[n]  = feq_W[n];
      fi_S[n]  = feq_S[n];
      fi_N[n]  = feq_N[n];
      fi_SE[n] = feq_SE[n];
      fi_NE[n] = feq_NE[n];
      fi_SW[n] = feq_SW[n];
      fi_NW[n] = feq_NW[n];
    }
    else if (map[gi] == GEO_WALL) {
      float t;
      t = fi_E[n];
      fi_E[n] = fi_W[n];
      fi_W[n] = t;

      t = fi_NW[n];
      fi_NW[n] = fi_SE[n];
      fi_SE[n] = t;

      t = fi_NE[n];
      fi_NE[n] = fi_SW[n];
      fi_SW[n] = t;

      t = fi_N[n];
      fi_N[n] = fi_S[n];
      fi_S[n] = t;
    }
  }

  for (n = 0; n < N_FLUID; n++) {
    od[n].fC[gi] = fi_C[n];

    // N + S propagation (global memory)
    if (blockIdx.y > 0)          od[n].fS[gi - LAT_W] = fi_S[n];
    if (blockIdx.y < LAT_H - 1)  od[n].fN[gi + LAT_W] = fi_N[n];

    // E propagation in shared memory
    if (tix < blockDim.x - 1) {
      fo_E[tix + 1] = fi_E[n];
      fo_NE[tix + 1] = fi_NE[n];
      fo_SE[tix + 1] = fi_SE[n];
      // E propagation in global memory (at block boundary)
    }
    else if (ti < LAT_W) {
      od[n].fE[gi + 1] = fi_E[n];
      if (blockIdx.y > 0)          od[n].fSE[gi-LAT_W + 1] = fi_SE[n];
      if (blockIdx.y < LAT_H - 1)  od[n].fNE[gi+LAT_W + 1] = fi_NE[n];
    }

    // W propagation in shared memory
    if (tix > 0) {
      fo_W[tix - 1] = fi_W[n];
      fo_NW[tix - 1] = fi_NW[n];
      fo_SW[tix - 1] = fi_SW[n];
    // W propagation in global memory (at block boundary)
    }
    else if (ti > 0) {
      od[n].fW[gi - 1] = fi_W[n];
      if (blockIdx.y > 0)         od[n].fSW[gi - LAT_W - 1] = fi_SW[n];
      if (blockIdx.y < LAT_H - 1) od[n].fNW[gi + LAT_W - 1] = fi_NW[n];
    }

    __syncthreads();

    // The leftmost thread is not updated in this block
    if (tix > 0) {
      od[n].fE[gi] = fo_E[tix];
      if (blockIdx.y > 0)          od[n].fSE[gi - LAT_W] = fo_SE[tix];
      if (blockIdx.y < LAT_H - 1)  od[n].fNE[gi + LAT_W] = fo_NE[tix];
    }

    // The rightmost thread is not updated in this block
    if (tix < blockDim.x - 1) {
      od[n].fW[gi] = fo_W[tix];
      if (blockIdx.y > 0)          od[n].fSW[gi - LAT_W] = fo_SW[tix];
      if (blockIdx.y < LAT_H - 1)  od[n].fNW[gi + LAT_W] = fo_NW[tix];
    }
  }

}

//------------------------------------------------------------------------------
void cuCollidePropagate(void)
{
  unsigned int n;
  dim3 grid;

  grid.x = LAT_W / BLOCK_SIZE;
  grid.y = LAT_H;

  for (n=0; n < 400; n++) {
    if (n % 299 == 0) {
      //TODO: Do not pass dev_state->.... pointer
      knCollidePropogate <<<grid,BLOCK_SIZE>>>
        (state.dev_map, dev_state->dev_dist2, dev_state->dev_dist1,
         state.dev_rho, state.dev_vx, state.dev_vy);
    }
    else if (n % 2 == 0) {
      knCollidePropogate <<<grid,BLOCK_SIZE>>>
        (state.dev_map, dev_state->dev_dist1, dev_state->dev_dist2,
         NULL, NULL, NULL);
    }
    else {
      knCollidePropogate <<<grid,BLOCK_SIZE>>>
        (state.dev_map, dev_state->dev_dist2, dev_state->dev_dist1,
         NULL, NULL, NULL);
    }
  }
}

//------------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  double *mx_param;
  float *param;
  unsigned int n, m;

  // Check for proper number of arguments
  if (nrhs != 1) {
    mexErrMsgTxt("One input required.");
  } else if (nlhs > 3) {
    mexErrMsgTxt("Too many output arguments.");
  }

  if (!readState(prhs)) mexErrMsgTxt("Invalid CUDA handle.");

  cuCollidePropagate();

  if (nlhs >= 1) {
   plhs[0] = mxCreateDoubleMatrix(LAT_W, LAT_H, mxREAL);
   mx_param = mxGetPr(plhs[0]);
   param = (float *) mxMalloc(SIZE_F);

   cudaMemcpy(param, state.dev_vx, SIZE_F, cudaMemcpyDeviceToHost);

   for (n = 0; n < LAT_W; n++) {
     for (m = 0; m < LAT_H; m++) {
       mx_param[n + m * LAT_W] = (double) param[n + m * LAT_W];
     }
   }
   mxFree(param);
  }

  if (nlhs >= 2) {
   plhs[1] = mxCreateDoubleMatrix(LAT_W, LAT_H, mxREAL);
   mx_param = mxGetPr(plhs[1]);
   param = (float *) mxMalloc(SIZE_F);

   cudaMemcpy(param, state.dev_vy, SIZE_F, cudaMemcpyDeviceToHost);

   for (n = 0; n < LAT_W; n++) {
     for (m = 0; m < LAT_H; m++) {
       mx_param[n + m * LAT_W] = (double) param[n + m * LAT_W];
     }
   }
   mxFree(param);
  }

  if (nlhs >= 3) {
   plhs[2] = mxCreateDoubleMatrix(LAT_W, LAT_H, mxREAL);
   mx_param = mxGetPr(plhs[2]);
   param = (float *)mxMalloc(SIZE_F);

   cudaMemcpy(param, state.dev_rho, SIZE_F, cudaMemcpyDeviceToHost);

   for (n = 0; n < LAT_W; n++) {
     for (m = 0; m < LAT_H; m++) {
       mx_param[n + m * LAT_W] = (double) param[n + m * LAT_W];
     }
   }
   mxFree(param);
  }

  syncState();
}

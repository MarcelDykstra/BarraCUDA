#ifndef __BARRACUDA_H
#define __BARRACUDA_H

#include <cuda.h>
#include <mex.h>

#define GEO_FLUID    0
#define GEO_WALL     1
#define GEO_INFLOW   2

#define GRID_SIZE_W  16
#define GRID_SIZE_D  16
#define GRID_SIZE_H  16

#define BLOCK_SIZE_W 4
#define BLOCK_SIZE_D 4
#define BLOCK_SIZE_H 4

#define LAT_SIZE_W  (GRID_SIZE_W * BLOCK_SIZE_W)
#define LAT_SIZE_D  (GRID_SIZE_D * BLOCK_SIZE_D)
#define LAT_SIZE_H  (GRID_SIZE_H * BLOCK_SIZE_H)

#define SIZE_I  (LAT_SIZE_W * LAT_SIZE_D * LAT_SIZE_H * sizeof(unsigned int))
#define SIZE_F  (LAT_SIZE_W * LAT_SIZE_D * LAT_SIZE_H * sizeof(float))

#define GBL_IDX(w, d, h)  ((w) + (d) * LAT_SIZE_W + \
                           (h) * LAT_SIZE_W * LAT_SIZE_D)
#define BLK_IDX(w, d, h)  ((w) + (d) * BLOCK_SIZE_W + \
                           (h) * BLOCK_SIZE_W * BLOCK_SIZE_H)

#define MOD(x,y)  (x % y + y) % y

#define N_FLUID   2

#define VISC_1    0.01
#define TAU_1     ((6.0 * VISC_1 + 1.0) / 2.0)

#define VISC_2    0.04
#define TAU_2     ((6.0 * VISC_2 + 1.0) / 2.0)

// D3Q19 unit cell
// C=centre, N=north, S=south, W=west, E=east, U=up, D=down

typedef struct {
    float fE, fW, fS, fN;
    float fSE, fSW, fNE, fNW;
    float fU, fUE, fUW, fUS, fUN;
    float fD, fDE, fDW, fDS, fDN;
} propa_t; // In-block propagation

typedef struct {
    float fC;
    float fE, fW, fS, fN;
    float fSE, fSW, fNE, fNW;
    float fU, fUE, fUW, fUS, fUN;
    float fD, fDE, fDW, fDS, fDN;
} dist_t;

typedef struct {
    float *fC;
    float *fE, *fW, *fS, *fN;
    float *fSE, *fSW, *fNE, *fNW;
    float *fU, *fUE, *fUW, *fUS, *fUN;
    float *fD, *fDE, *fDW, *fDS, *fDN;
} dist_ptr_t;

typedef struct {
    unsigned int magic;
    unsigned int *dev_map;
    float *dev_vx, *dev_vy, *dev_vz, *dev_rho;
    dist_ptr_t dev_dist1[N_FLUID], dev_dist2[N_FLUID];
} state_t;

extern state_t state;
extern state_t *dev_state;

void createState(void);
void destroyState(void);
void writeState(mxArray *plhs[]);
void syncState(void);
bool readState(const mxArray *prhs[]);

#endif

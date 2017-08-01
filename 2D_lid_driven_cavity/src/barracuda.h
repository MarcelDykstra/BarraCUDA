#ifndef __BARRACUDA_H
#define __BARRACUDA_H

#include <cuda.h>
#include <mex.h>

#define GEO_FLUID   0
#define GEO_WALL    1
#define GEO_INFLOW  2

#define LAT_H       256
#define LAT_W       256
#define BLOCK_SIZE  64
#define N_FLUID     1

#define SIZE_I      (LAT_W * LAT_H * sizeof(unsigned int))
#define SIZE_F      (LAT_W * LAT_H * sizeof(float))

#define VISC_1      0.01
#define TAU_1       ((6.0 * VISC_1 + 1.0) / 2.0)

#define VISC_2      0.04
#define TAU_2       ((6.0 * VISC_2 + 1.0) / 2.0)


typedef struct {
    float *fC, *fE, *fW, *fS, *fN, *fSE, *fSW, *fNE, *fNW;
} dist_t;

typedef struct {
    unsigned int magic;
    unsigned int *dev_map;
    float *dev_vx, *dev_vy, *dev_rho;
    dist_t dev_dist1[N_FLUID], dev_dist2[N_FLUID];
} state_t;

extern state_t state;
extern state_t *dev_state;

void createState(void);
void destroyState(void);
void writeState(mxArray *plhs[]);
void syncState(void);
bool readState(const mxArray *prhs[]);

#endif

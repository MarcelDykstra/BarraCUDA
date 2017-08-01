#include <stdio.h>
#include <string.h>
#include <mex.h>
#include <cuda.h>

#define NUMBER_OF_FIELDS (sizeof(field_names) / sizeof(*field_names))

//------------------------------------------------------------------------------
void cudaDeviceProperties(cudaDeviceProp *prop)
{
    int count;
    cudaGetDeviceCount(&count);
    for (int i=0; i < count; i++) {
        cudaGetDeviceProperties(prop, i);
    }
}

//------------------------------------------------------------------------------
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    cudaDeviceProp prop;
    const char *field_names[] = {
        "name", "totalGlobalMem", "sharedMemPerBlock", "regsPerBlock",
        "warpSize", "memPitch", "maxThreadsPerBlock", "maxThreadsDim",
        "maxGridSize", "totalConstMem", "major", "minor", "clockRate",
        "textureAlignment", "deviceOverlap", "multiProcessorCount",
        "kernelExecTimeoutEnabled", "integrated", "canMapHostMemory",
        "computeMode", "maxTexture1D", "maxTexture2D", "maxTexture3D",
        "concurentKernels"
    };
    mwSize dims[2] = {1, 1};
    mxArray *mtrx;
    double *elem;

    if (nrhs != 0) mexErrMsgTxt("No input argument required.");
    if (nlhs > 1) mexErrMsgTxt("Too many output arguments.");

    cudaDeviceProperties(&prop);

    plhs[0] = mxCreateStructArray(2, dims, NUMBER_OF_FIELDS, field_names);

    mxSetFieldByNumber(plhs[0], 0, 0,
                       mxCreateString(prop.name));
    mxSetFieldByNumber(plhs[0], 0, 1,
                       mxCreateDoubleScalar(prop.totalGlobalMem));
    mxSetFieldByNumber(plhs[0], 0, 2,
                       mxCreateDoubleScalar(prop.sharedMemPerBlock));
    mxSetFieldByNumber(plhs[0], 0, 3,
                       mxCreateDoubleScalar(prop.regsPerBlock));
    mxSetFieldByNumber(plhs[0], 0, 4,
                       mxCreateDoubleScalar(prop.warpSize));
    mxSetFieldByNumber(plhs[0], 0, 5,
                       mxCreateDoubleScalar(prop.memPitch));
    mxSetFieldByNumber(plhs[0], 0, 6,
                       mxCreateDoubleScalar(prop.maxThreadsPerBlock));
    mtrx = mxCreateDoubleMatrix(1, 3, mxREAL);
    elem = mxGetPr(mtrx);
    elem[0] = prop.maxThreadsDim[0];
    elem[1] = prop.maxThreadsDim[1];
    elem[2] = prop.maxThreadsDim[2];
    mxSetFieldByNumber(plhs[0], 0, 7, mtrx);
    mtrx = mxCreateDoubleMatrix(1, 3, mxREAL);
    elem = mxGetPr(mtrx);
    elem[0] = prop.maxGridSize[0];
    elem[1] = prop.maxGridSize[1];
    elem[2] = prop.maxGridSize[2];
    mxSetFieldByNumber(plhs[0], 0, 8, mtrx);
    mxSetFieldByNumber(plhs[0], 0, 9,
                       mxCreateDoubleScalar(prop.totalConstMem));
    mxSetFieldByNumber(plhs[0], 0, 10,
                       mxCreateDoubleScalar(prop.major));
    mxSetFieldByNumber(plhs[0], 0, 11,
                       mxCreateDoubleScalar(prop.minor));
    mxSetFieldByNumber(plhs[0], 0, 12,
                       mxCreateDoubleScalar(prop.clockRate));
    mxSetFieldByNumber(plhs[0], 0, 13,
                       mxCreateDoubleScalar(prop.textureAlignment));
    mxSetFieldByNumber(plhs[0], 0, 14,
                       mxCreateLogicalScalar(prop.deviceOverlap));
    mxSetFieldByNumber(plhs[0], 0, 15,
                       mxCreateDoubleScalar(prop.multiProcessorCount));
    mxSetFieldByNumber(plhs[0], 0, 16,
                       mxCreateLogicalScalar(prop.kernelExecTimeoutEnabled));
    mxSetFieldByNumber(plhs[0], 0, 17,
                       mxCreateLogicalScalar(prop.integrated));
    mxSetFieldByNumber(plhs[0], 0, 18,
                       mxCreateLogicalScalar(prop.canMapHostMemory));
    if (prop.computeMode == 0)
        mxSetFieldByNumber(plhs[0],0,19,mxCreateString("default"));
    else if (prop.computeMode == 1)
        mxSetFieldByNumber(plhs[0],0,19,mxCreateString("exclusive"));
    else if (prop.computeMode == 2)
        mxSetFieldByNumber(plhs[0],0,19,mxCreateString("prohibited"));
                           mxSetFieldByNumber(plhs[0], 0, 20,
                           mxCreateDoubleScalar(prop.maxTexture1D));
    mtrx = mxCreateDoubleMatrix(1, 2, mxREAL);
    elem = mxGetPr(mtrx);
    elem[0] = prop.maxTexture2D[0];
    elem[1] = prop.maxTexture2D[1];
    mxSetFieldByNumber(plhs[0], 0, 21,mtrx);
    mtrx = mxCreateDoubleMatrix(1, 3, mxREAL);
    elem = mxGetPr(mtrx);
    elem[0] = prop.maxTexture3D[0];
    elem[1] = prop.maxTexture3D[1];
    elem[2] = prop.maxTexture3D[2];
    mxSetFieldByNumber(plhs[0], 0, 22,mtrx);
    ///mtrx = mxCreateDoubleMatrix(1, 3, mxREAL);
    ///elem = mxGetPr(mtrx);
    ///elem[0] = prop.maxTexture2DArray[0];
    ///elem[1] = prop.maxTexture2DArray[1];
    ///elem[2] = prop.maxTexture2DArray[2];
    ///mxSetFieldByNumber(plhs[0],0,23,mtrx);
    mxSetFieldByNumber(plhs[0], 0, 23,
                       mxCreateLogicalScalar(prop.concurrentKernels));
}

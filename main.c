/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "translation.h"
#include "structs.h"
#include "integral.h"
#include "octree.h"

extern float INTPT_h[INTORDER];
extern float INTWGT_h[INTORDER];


int main (int argc, char** argv)
{
    CUDA_CALL(cudaDeviceReset());
    CUDA_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,0.01*1024*1024*1024));
    
    size_t totalMem, freeMem;
    CUDA_CALL(cudaMemGetInfo(&freeMem,&totalMem));
    printf("Available memory: %fGB\n",(float)freeMem/(1024*1024*1024));
    
    int p1 = 5, p2 = 2;
    float t = 2.3, wavNum = 13.1;
    cuFloatComplex *sparseMat1 = (cuFloatComplex*)calloc(sparseCoaxTransSize(p1),sizeof(cuFloatComplex));
    genSRSparseCoaxTransMat(wavNum,&t,1,p1,sparseMat1);
    
    
    cuFloatComplex *sparseMat2 = (cuFloatComplex*)calloc(sparseCoaxTransSize(p2),sizeof(cuFloatComplex));
    genSRSparseCoaxTransMat(wavNum,&t,1,p2,sparseMat2);
    printMat_cuFloatComplex(sparseMat2,1,sparseCoaxTransSize(p2),1);
    
    reduceSparseCoaxTransMat(sparseMat1,p1,sparseMat2,p2);
    printMat_cuFloatComplex(sparseMat2,1,sparseCoaxTransSize(p2),1);
    
    
    free(sparseMat1);
    free(sparseMat2);
    
    return EXIT_SUCCESS;
}


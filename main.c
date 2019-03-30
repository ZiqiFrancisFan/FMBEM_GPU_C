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


int main (int argc, char** argv)
{
    CUDA_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,512*1024*1024));
    int p = 3;
    float t = 5.3, wavNum = 9.5;
    cuFloatComplex *H = (cuFloatComplex*)malloc((2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex));
    cuFloatComplex *denseMat = (cuFloatComplex*)malloc(p*p*p*p*sizeof(cuFloatComplex));
    cuFloatComplex *sparseMat = (cuFloatComplex*)malloc(p*(2*p*p+3*p+1)/6*sizeof(cuFloatComplex));
    
    srCoaxTransMatsInit(wavNum,&t,1,p,H);
    coaxTransMatGen(H,p,denseMat);
    getSparseMatFromCoaxTransMat(denseMat,p,sparseMat);
    printf("Dense to sparse: \n");
    printMat_cuFloatComplex(sparseMat,1,p*(2*p*p+3*p+1)/6,1);
    
    srCoaxTransMatsInit(wavNum,&t,1,p,H);
    sparseCoaxTransMatGen(H,p,sparseMat);
    printf("Sparse: \n");
    printMat_cuFloatComplex(sparseMat,1,p*(2*p*p+3*p+1)/6,1);
    
    free(H);
    free(denseMat);
    free(sparseMat);
    
    return EXIT_SUCCESS;
}


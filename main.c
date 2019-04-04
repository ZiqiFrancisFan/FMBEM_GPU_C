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
    CUDA_CALL(cudaDeviceReset());
    CUDA_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,0.01*1024*1024*1024));
    
    size_t totalMem, freeMem;
    CUDA_CALL(cudaMemGetInfo(&freeMem,&totalMem));
    printf("Available memory: %fGB\n",(float)freeMem/(1024*1024*1024));
    
    octree oct;
    initOctree(&oct);
    printf("Successfully initialized octree.\n");
    genOctree("sphere_10mm.obj",300,1,&oct);
    printf("successfully generated octree.\n");
    printf("Number of RR translations: %d\n",oct.numRRCoaxTransVec);
    printFloatArray(oct.rrCoaxTransVec,oct.numRRCoaxTransVec);
    printf("Number of SR translations: %d\n",oct.numSRCoaxTransVec);
    printFloatArray(oct.srCoaxTransVec,oct.numSRCoaxTransVec);
    printf("Number of rotation angles: %d\n",oct.numRotAng);
    printRotAngArray(oct.ang,oct.numRotAng);
    printFMMLevelSet(oct.fmmLevelSet,oct.lmax);
    destroyOctree(&oct);
    
    printf("successfully destroyed octree.\n");
    return EXIT_SUCCESS;
}


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
    
    octree oct;
    initOctree(&oct);
    printf("Successfully initialized octree.\n");
    genOctree("sphere_10mm.obj",200,1,&oct);
    printf("successfully generated octree.\n");
    printf("Number of RR translations: %d\n",oct.numRRCoaxTransVec);
    printFloatArray(oct.rrCoaxTransVec,oct.numRRCoaxTransVec);
    printf("Number of SR translations: %d\n",oct.numSRCoaxTransVec);
    printFloatArray(oct.srCoaxTransVec,oct.numSRCoaxTransVec);
    printf("Number of rotation angles: %d\n",oct.numRotAng);
    printRotAngArray(oct.ang,oct.numRotAng);
    
    //printf("Bottom element indices: \n");
    //printIntArray(oct.btmLvlElemIdx,oct.numElem);
    printSSLevelTransIdxArr(oct.ssTransIdx,oct.lmax,oct.fmmLevelSet);
    printSSLevelTransDest(oct.ssTransDestArr,oct.lmax,oct.fmmLevelSet);
    printSRLevelTransIdxArr(oct.srTransIdx,oct.lmax,oct.srNumLevelArr,oct.fmmLevelSet);
    printSRLevelTransOrigin(oct.srTransOriginArr,oct.lmax,oct.srNumLevelArr,oct.fmmLevelSet);
    printFMMLevelSet(oct.fmmLevelSet,oct.lmax);
    printSRLevelTransDest(oct.srTransDestArr,oct.lmax,oct.srNumLevelArr,oct.fmmLevelSet);
    
    printRRLevelTransIdxArr(oct.rrTransIdx,oct.lmax,oct.fmmLevelSet);
    printRRLevelTransOrigin(oct.rrTransOriginArr,oct.lmax,oct.fmmLevelSet);
    /*
    for(int i=0;i<oct.numRotAng;i++) {
        printf("%dth rotation matrix: \n",i);
        printMat_cuFloatComplex(&(oct.rotMat1[i*oct.pmax*(4*oct.pmax*oct.pmax-1)/3]),1,
                oct.pmax*(4*oct.pmax*oct.pmax-1)/3,1);
    }
     */
    printf("element index: \n");
    printIntArray(oct.btmLvlElemIdx,oct.numElem);
    
    destroyOctree(&oct);
    
    printf("successfully destroyed octree.\n");
    return EXIT_SUCCESS;
}


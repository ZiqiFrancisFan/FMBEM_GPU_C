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
    
    float *vec;
    rotAng *angle;
    int numPt, numElem;
    findNum("sphere_300mm.obj",&numPt,&numElem);
    cartCoord_d *pt = (cartCoord_d*)malloc(numPt*sizeof(cartCoord_d));
    triElem *elem = (triElem*)malloc(numElem*sizeof(triElem));
    readOBJ("sphere_300mm.obj",pt,elem);
    int *srcBoxSet = (int*)malloc((numElem+1)*sizeof(int));
    int lmax, numVec, numRotAngle;
    double d;
    cartCoord_d pt_min;
    srcBoxes(pt,elem,numElem,1,srcBoxSet,&lmax,&d,&pt_min);
    genRRCoaxTransVecsRotAngles(3,d,pt_min,&vec,&numVec,&angle,&numRotAngle);
    printf("numVec: %d, numRotAngle: %d\n",numVec,numRotAngle);
    printRotAngArray(angle,numRotAngle);
    for(int i=0;i<numVec;i++) {
        printf("%f ",vec[i]);
    }
    printf("\n");
    float p = truncNum(310,0.05,1.5,pow(2,-2)*d);
    printf("p=%f\n",p);
    float memNeed = (p*(2*p*p+3*p+1)/6*(float)numVec+p*(4*p*p-1)/3*2*(float)numRotAngle)*8.0f/(1024.0f*1024.0f*1024.0f);
    printf("Largest memory need: %fGB\n",memNeed);
    
    bubbleSort(vec,numVec);
    for(int i=0;i<numVec;i++) {
        printf("%f ",vec[i]);
    }
    printf("\n");
    
    rotAng *tempAngle = (rotAng*)malloc(numRotAngle*sizeof(rotAng));
    for(int i=0;i<numRotAngle;i++) {
        tempAngle[i] = angle[i];
    }
    sortRotArray(angle,numRotAngle,0.00000001);
    printf("sorted rots: \n");
    printRotAngArray(angle,numRotAngle);
    if(equalRotArrays(angle,tempAngle,numRotAngle)) {
        printf("Equal.\n");
    }
    
    
    
    free(vec);
    free(angle);
    free(tempAngle);
    free(pt);
    free(elem);
    free(srcBoxSet);
    
    return EXIT_SUCCESS;
}


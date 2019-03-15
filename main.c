/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <stdio.h>
#include <stdlib.h>
#include "translation.h"
#include "structs.h"
#include "integral.h"


int main (int argc, char** argv)
{
    float *pt = (float*)malloc(INTORDER*sizeof(float));
    float *wgt = (float*)malloc(INTORDER*sizeof(float));
    HOST_CALL(genGaussParams(INTORDER,pt,wgt));
    cartCoord nod[3], y;
    nod[0] = {2.3,3.2,0.9};
    nod[1] = {-0.3,-0.9,-0.4};
    nod[2] = {0,0.1,-0.1};
    y = {4,1,2};
    float wavNum = 9.3;
    cartCoord nrml = {1,2,3};
    nrml = normalize(nrml);
    cuFloatComplex z = triElemIntegral_p2Gpn1pn2_sgl(wavNum,nod,pt,wgt);
    printMat_cuFloatComplex(&z,1,1,1);
    z = triElemIntegral_pRpn(wavNum,nod,0,0,y,pt,wgt);
    printMat_cuFloatComplex(&z,1,1,1);
    free(pt);
    free(wgt);
    return EXIT_SUCCESS;
}


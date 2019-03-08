/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <stdio.h>
#include <stdlib.h>
#include "numerical.h"
#include "structs.h"


int main (int argc, char** argv)
{
    int p = 4;
    //rotAng rang = {0.3*PI,0.2*PI,0.4*PI};
    cartCoord vec = {1.2,-0.3,2.2};
    float wavNum = 50.3;
    cuFloatComplex *coeff = (cuFloatComplex*)malloc(p*p*sizeof(cuFloatComplex));
    HOST_CALL(genRndCoeffs(p*p,coeff));
    //printMat_cuFloatComplex(coeff,1,p*p,1);
    cuFloatComplex *prod = (cuFloatComplex*)malloc(p*p*sizeof(cuFloatComplex));
    HOST_CALL(transMatsVecsMul_SR(wavNum,&vec,coeff,1,p,prod));
    printMat_cuFloatComplex(prod,1,p*p,1);
    HOST_CALL(transMatsVecsMul_SR_rcr(wavNum,&vec,coeff,1,p,prod));
    printMat_cuFloatComplex(prod,1,p*p,1);
    free(coeff);
    free(prod);
}


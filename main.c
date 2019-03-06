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
    int p = 3;
    cartCoord vec = {0.3,0.2,0.3};
    float wavNum = 9.7;
    cuFloatComplex *mat = (cuFloatComplex*)malloc(p*p*p*p*sizeof(cuFloatComplex));
    genTransMat(wavNum,&vec,1,p,mat);
    printMat_cuFloatComplex(mat,p*p,p*p,p*p);
    free(mat);
}


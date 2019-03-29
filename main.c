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
    float transVec[2];
    float wavNum = 10.5;
    transVec[0] = 4.3;
    transVec[1] = 3.2;
    
    int p = 3;
    
    HOST_CALL(testSparseCoaxTransMatsGen(wavNum,transVec,2,p));
    
    return EXIT_SUCCESS;
}


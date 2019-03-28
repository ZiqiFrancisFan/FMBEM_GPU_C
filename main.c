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
    octree oct;
    printf("%d\n",3/3);
    genOctree("Head_20kHz.obj",9.3,1,&oct);
    destroyOctree(&oct,oct.lmax);
    return EXIT_SUCCESS;
}


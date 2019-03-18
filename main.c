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
    cartCoord_d *pt;
    triElem *elem;
    int numPt, numElem;
    findNum("cube_12.obj",&numPt,&numElem);
    pt = (cartCoord_d*)malloc(numPt*sizeof(cartCoord_d));
    elem = (triElem*)malloc(numElem*sizeof(triElem));
    readOBJ("cube_12.obj",pt,elem);
    int lmax;
    int *srcBoxSet = (int*)malloc((numElem+1)*sizeof(int));
    double d;
    cartCoord_d pt_min;
    srcBoxes(pt,elem,numElem,2,srcBoxSet,&lmax,&d,&pt_min);
    printf("d=%f\n",d);
    printf("Maximum level: %d\n",lmax);
    printSet(srcBoxSet);
    printPnts(&pt_min,1);
    free(pt);
    free(elem);
    free(srcBoxSet);
    return EXIT_SUCCESS;
}


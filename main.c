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
    int p = truncNum(9.3,0.01,1.5,0.3/16);
    printf("p=%d\n",p);
    return EXIT_SUCCESS;
}


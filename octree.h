/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   octree.h
 * Author: ziqi
 *
 * Created on March 5, 2019, 9:15 AM
 */

#ifndef OCTREE_H
#define OCTREE_H

#ifdef __cplusplus
extern "C" {
#endif
#include "structs.h"
#include <stdbool.h>
#ifndef NUM_BITs_BIN
#define NUM_BITs_BIN (sizeof(unsigned))
#endif
    
#ifndef MAX
#define MAX 100000
#endif

    void findNum(const char *filename, int *pV, int *pE);

    int readOBJ(const char *filename, cartCoord_d *p, triElem *e);

    void printSet(const int *set);

    bool arrEqual(const int *a, const int *b, const int num);

    int parent(int num);

    int child(int num, int cld);

    void children(const int num, int *cldrn);

    cartCoord_d scale(const cartCoord_d x, const cartCoord_d x_min, const double d);

    cartCoord_d descale(const cartCoord_d x_s, const cartCoord_d x_min, const double d);

    double descale_1d(const double a, const double D, const double v_min);


    void scalePnts(const cartCoord_d* pnt, const int numPnts, const cartCoord_d pnt_min, 
            const double d, cartCoord_d* pnt_scaled);

    void dec2bin_frac(double s, int l, int *h);

    void dec2bin_int(unsigned num, int *rep, int *numBits);

    void bitIntleave(const int *x, const int *y, const int *z, const int l, int *result);

    void bitDeintleave(const int *result, const int l, int *x, int *y, int *z);

    int indArr2num(const int *ind, const int l, const int d);

    int pnt2boxnum(const cartCoord_d pnt, const int l);

    cartCoord_d boxCenter(const int num, const int l);

    int neighbors(const int num, const int l, int *numNeighbors, int *nbr);

    void createSet(const int *elems, const int numElems, int *set);

    bool isMember(const int t, const int *set);

    bool isEmpty(const int *set);

    void intersection(const int *set1, const int *set2, int *set3);

    void Union(const int *set1, const int *set2, int *set3);

    void difference(const int *set1, const int *set2, int *set3);

    void pnts2numSet(const cartCoord_d *pnts, const int numPnts, const int l, 
            int *set);

    void sampleSpace(const int l, int *set);

    void I1(const int num, int *set);

    void I2(const int num, const int l, int *set);

    void I3(const int num, const int l, int *set);

    //applicable to levels larger than or equal to 2
    void I4(const int num, const int l, int *set);

    void orderArray(const int *a, const int num, int *ind);

    void printPnts(const cartCoord_d *p, const int numPnts);

    void printElems(const triElem *e, const int numElems);
    
    void genOctPt(const int level, cartCoord_d *pt);

    int deterLmax(const cartCoord_d *pnts, const int numPnts, const int s);

    void findBoundingCube(const cartCoord_d *pnts, const int numPnts, const double eps, 
            cartCoord_d *pnts_b, double *d);

    void srcBoxes(const cartCoord_d *pnts, const triElem *elems, const int numElems, 
            const int s, int *srcBoxSet, int *lmax, double *D, cartCoord_d *pnt_min);

    int truncNum(const double k, const double eps, const double sigma, const double a);

    int truncNum_2(const double wavNum, const double eps, const double sigma, const double a);

    void prntLevelSet(const int *X, const int l, int *X_n);

    void FMMLvlSet_s(const int *X, const int lmax, int ***pSet);

    void FMMLvlSet_e(const int *Y, const int lmax, int ***pSet);
    
    void FMMLevelSet(const int *btmLvl, const int lmax, int **pSet);

    int findSetInd(const int *X, const int num);

    void sortSet(int *set);
    
    void findNum(const char *filename, int *pV, int *pE);
    
    int readOBJ(const char *filename, cartCoord_d *p, triElem *e);
    
    void FMMLevelSetNumSR(int **pSet, const int lmax, int **numSet);
    
    void printLevelSetNumSR(int **numSet, int **pSet, const int lmax);
    
    void printFMMLevelSet(int **pSet, const int lmax);

#ifdef __cplusplus
}
#endif

#endif /* OCTREE_H */


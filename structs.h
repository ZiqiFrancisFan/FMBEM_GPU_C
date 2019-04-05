/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   structs.h
 * Author: ziqi
 *
 * Created on March 5, 2019, 9:13 AM
 */

#ifndef STRUCTS_H
#define STRUCTS_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include <cuComplex.h>
    
#ifndef PI
#define PI 3.1415926535897932f
#endif

#define IDXC0(row,column,stride) ((column)*(stride)+(row))

    struct triElem
    {
        int node[3]; //the three nodes on the triangular element
        cuFloatComplex alpha;
        cuFloatComplex beta;
        cuFloatComplex gamma;
    };

    struct cartCoord
    {
        float x;
        float y;
        float z;
    };
    
    struct cartCoord_double
    {
        double x;
        double y;
        double z;
    };
    
    struct sphCoord
    {
        float r;
        float theta;
        float phi;
    };
    
    struct sphCoord_double
    {
        double r;
        double theta;
        double phi;
    };
    
    struct rotAng
    {
        float alpha;
        float beta;
        float gamma;
    };
    
    //includes the index for the coaxial translation and the index for the rotation angle
    struct transIdx
    {
        int coaxIdx;
        int rotIdx;
    };
    
    typedef struct triElem triElem;
    
    typedef struct cartCoord cartCoord;
    
    typedef struct cartCoord_double cartCoord_d;
    
    typedef struct sphCoord sphCoord;
    
    typedef struct sphCoord_double sphCoord_d;
    
    typedef struct rotAng rotAng;
    
    typedef struct transIdx transIdx;
    
    struct octree
    {
        //points and elements
        cartCoord *pt;
        triElem *elem;
        
        //number of points and elements
        int numPt;
        int numElem;
        
        //level sets
        int **fmmLevelSet;
        int lmax;
        int lmin;
        double d;
        cartCoord_d pt_min;
        
        //sparse matrices for ss, sr and rr translations, saved on host memory
        cuFloatComplex *rotMat1;
        cuFloatComplex *rotMat2;
        
        cuFloatComplex *srCoaxMat;
        cuFloatComplex *rrCoaxMat;
        
        float *srCoaxTransVec;
        int numSRCoaxTransVec;
        float *rrCoaxTransVec;
        int numRRCoaxTransVec;
        rotAng *ang;
        int numRotAng;
        
        float eps;
        float maxWavNum;
        
        int *btmLvlElemIdx;
        
        //indices for translation vectors
        transIdx **ssTransIdx;
        transIdx **srTransIdx;
        transIdx **rrTransIdx;
    };
    
    typedef struct octree octree;
    
    __host__ __device__ float dotProd(const cartCoord u, const cartCoord v);

    __host__ __device__ cartCoord scalarMul(const float lambda, const cartCoord v);

    __host__ __device__ cartCoord crossProd(const cartCoord u, const cartCoord v);

    __host__ __device__ cartCoord cartCoordAdd(const cartCoord u, const cartCoord v);

    __host__ __device__ cartCoord cartCoordSub(const cartCoord u, const cartCoord v);
    
    __host__ __device__ float cartNorm(const cartCoord x);

    __host__ __device__ float cartDist(const cartCoord x, const cartCoord y);
    
    __host__ __device__ cartCoord normalize(const cartCoord x);

    __host__ __device__ sphCoord cart2sph(const cartCoord s);
    
    __host__ __device__ sphCoord_d cart_d2sph_d(const cartCoord_d s);
    
    __host__ __device__ sphCoord sphCoord_d2sphCoord(const sphCoord_d s);

    __host__ __device__ cartCoord sph2cart(const sphCoord s);
    
    __host__ __device__ cartCoord_d cartCoordSub_d(const cartCoord_d u, const cartCoord_d v);
    
    __host__ __device__ cartCoord_d triCentroid_d(const cartCoord_d nod[3]);
    
    __host__ __device__ cartCoord cartCoord_d2cartCoord(const cartCoord_d s);
    
    __host__ __device__ cartCoord_d cartCoord2cartCoord_d(const cartCoord s);
    
    __host__ void printCartCoordArray_d(const cartCoord_d *arr, const int num);
    
    __host__ void printCartCoordArray(const cartCoord *arr, const int num);
    
    __host__ void printSphCoordArray(const sphCoord *arr, const int num);
    
    __host__ void printFloatArray(const float *arr, const int num);
    
    __host__ void printIntArray(const int *arr, const int num);
    
    __host__ void printRotAngArray(const rotAng *angle, const int numAng);
    
    __host__ bool equalRotArrays(const rotAng *ang1, const rotAng *ang2, const int numAng);
    
    //__host__ void sortRotArray(rotAng *angle, const int numRot);
    
    __host__ void bubbleSort(float *arr, const int n);
    
    __host__ void sortRotArray(rotAng *ang, const int numRot, const float eps);

#ifdef __cplusplus
}
#endif

#endif /* STRUCTS_H */


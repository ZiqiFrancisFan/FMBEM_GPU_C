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
    
    struct rotAng
    {
        float alpha;
        float beta;
        float gamma;
    };
    
    typedef struct triElem triElem;
    
    typedef struct cartCoord cartCoord;
    
    typedef struct cartCoord_double cartCoord_d;
    
    typedef struct sphCoord sphCoord;
    
    typedef struct rotAng rotAng;
    
    __host__ __device__ float dotProd(const cartCoord u, const cartCoord v);

    __host__ __device__ cartCoord scalarProd(const float lambda, const cartCoord v);

    __host__ __device__ cartCoord crossProd(const cartCoord u, const cartCoord v);

    __host__ __device__ cartCoord cartCoordAdd(const cartCoord u, const cartCoord v);

    __host__ __device__ cartCoord cartCoordSub(const cartCoord u, const cartCoord v);

    __host__ __device__ sphCoord cart2sph(const cartCoord s);

    __host__ __device__ cartCoord sph2cart(const sphCoord s);



#ifdef __cplusplus
}
#endif

#endif /* STRUCTS_H */


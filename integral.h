/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   integral.h
 * Author: ziqi
 *
 * Created on March 9, 2019, 8:51 AM
 */

#ifndef INTEGRAL_H
#define INTEGRAL_H

#ifndef INTORDER
#define INTORDER 3
#endif

#include "translation.h"

int genGaussParams(const int n, float *pt, float *wgt);

int gaussPtsToDevice(const float *pt, const float *wgt);

__host__ __device__ float cartNorm(const cartCoord x);

__host__ __device__ float cartDist(const cartCoord x, const cartCoord y);

__host__ __device__ cuFloatComplex green(const float k, const cartCoord x, const cartCoord y);



#endif /* INTEGRAL_H */


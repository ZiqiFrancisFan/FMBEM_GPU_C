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

__host__ __device__ cuFloatComplex green(const float k, const cartCoord x, const cartCoord y);

__host__ __device__ cuFloatComplex triElemIntegral_G_nsgl(const float wavNum, const cartCoord nod[3], const cartCoord y, 
        const float *pt, const float *wgt);

__host__ __device__ cuFloatComplex triElemIntegral_G_sgl(const float wavNum, const cartCoord nod[3], 
        const float *pt, const float *wgt);

__host__ cuFloatComplex dirDev_R(const float wavNum, const int n, const int m, 
        const cartCoord nrml, const cartCoord x);

__host__ cuFloatComplex triElemIntegral_pRpn(const float wavNum, const cartCoord nod[3], 
        const int n, const int m, const cartCoord x_lp, const float *pt, const float *wgt);

__host__ __device__ cuFloatComplex triElemIntegral_pGp1n_nsgl(const float wavNum, 
        const cartCoord nod[3], const cartCoord y, const float *pt, const float *wgt);

__host__ __device__ cuFloatComplex triElemIntegral_pGp1n_sgl(const float wavNum, const cartCoord nod[3], 
        const float *pt, const float *wgt);

__host__ __device__ cuFloatComplex triElemIntegral_pGp2n_nsgl(const float wavNum, 
        const cartCoord nod[3], const cartCoord nrml, const cartCoord y, 
        const float *pt, const float *wgt);

__host__ __device__ cuFloatComplex triElemIntegral_pGp2n_sgl(const float wavNum, 
        const cartCoord nod[3], const float *pt, const float *wgt);

__host__ __device__ cuFloatComplex triElemIntegral_p2Gp1np2n_nsgl(const float wavNum, 
        const cartCoord nod[3], const cartCoord nrml_y, const cartCoord y, 
        const float *pt, const float *wgt);

__host__ __device__ cuFloatComplex triElemIntegral_p2Gp1np2n_sgl(const float wavNum, 
        const cartCoord nod[3], const float *pt, const float *wgt);

__host__ __device__ void cmptDiffCoeff(const float wavNum, const cuFloatComplex *coeff, 
        const int p, const cartCoord nrml, cuFloatComplex *coeff_n);

#endif /* INTEGRAL_H */


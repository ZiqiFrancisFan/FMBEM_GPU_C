/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   numerical.h
 * Author: ziqi
 *
 * Created on March 5, 2019, 9:07 AM
 */

#ifndef NUMERICAL_H
#define NUMERICAL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
#include "device_launch_parameters.h"
#include "structs.h"


#ifndef PI
#define PI 3.1415926535897932f
#endif

#ifndef max
#define max(a,b) \
({ __typeof__ (a) _a = (a); \
__typeof__ (b) _b = (b); \
_a > _b ? _a : _b; })
#endif

#ifndef min
#define min(a,b) \
({ __typeof__ (a) _a = (a); \
__typeof__ (b) _b = (b); \
_a < _b ? _a : _b; })
#endif

#ifndef IDXC0
#define IDXC0(row,col,ld) ((ld)*(col)+(row))
#endif

#ifndef NM2IDX0
#define NM2IDX0(n,m) ((n)*(n)+(m)+(n))
#endif

#ifndef HOST_CALL
#define HOST_CALL(x) do {\
if((x)!=EXIT_SUCCESS){\
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)
#endif

#ifndef CUDA_CALL
#define CUDA_CALL(x) do {\
if((x)!=cudaSuccess) {\
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)
#endif

#ifndef CURAND_CALL
#define CURAND_CALL(x) do {\
if((x)!=CURAND_STATUS_SUCCESS) {\
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)
#endif

#ifndef CUBLAS_CALL
#define CUBLAS_CALL(x) \
do {\
if((x)!=CUBLAS_STATUS_SUCCESS)\
{\
printf("Error at %s:%d\n",__FILE__,__LINE__); \
if(x==CUBLAS_STATUS_NOT_INITIALIZED) { \
printf("The library was not initialized.\n"); \
}\
if(x==CUBLAS_STATUS_INVALID_VALUE) {\
printf("There were problems with the parameters.\n");\
}\
if(x==CUBLAS_STATUS_MAPPING_ERROR) {\
printf("There was an error accessing GPU memory.\n"); \
}\
return EXIT_FAILURE; } \
}\
while(0)
#endif

#ifndef CUSOLVER_CALL
#define CUSOLVER_CALL(x) \
do {\
if((x)!=CUSOLVER_STATUS_SUCCESS)\
{\
printf("Error at %s:%d\n",__FILE__,__LINE__); \
if((x)==CUSOLVER_STATUS_NOT_INITIALIZED) { \
printf("The library was not initialized.\n"); \
}\
if((x)==CUSOLVER_STATUS_INVALID_VALUE) {\
printf("Invalid parameters were passed.\n");\
}\
if((x)==CUSOLVER_STATUS_ARCH_MISMATCH) {\
printf("Achitecture not supported.\n"); \
}\
if((x)==CUSOLVER_STATUS_INTERNAL_ERROR) {\
printf("An internal operation failed.\n"); \
}\
return EXIT_FAILURE; } \
}\
while(0)
#endif

__host__ cuFloatComplex gsl_complex2cuFloatComplex(const gsl_complex rhs);

__host__ __device__ cuFloatComplex cplxExp(const float theta);

__host__ __device__ void cuMatMatMul(const cuFloatComplex *mat1, const cuFloatComplex *mat2, 
        const int numRow1, const int numCol1, const int numCol2, cuFloatComplex *mat);

__host__ gsl_complex gsl_sf_bessel_hl(const int l, const double s);

__host__ double factorial(const int n);

__host__ gsl_complex rglBasis(const double k, const int n, const int m, const sphCoord coord);

__host__ gsl_complex sglBasis(const double k, const int n, const int m, const sphCoord coord);

__host__ __device__ float aCoeff(const int n, const int m);

__host__ __device__ float bCoeff(const int n, const int m);

__host__ void rrTransMatsInit(const float wavNum, const cartCoord *vec, const int numVec, 
        const int p, cuFloatComplex *mat);

void printMat_cuFloatComplex(const cuFloatComplex* A, const int numRow, const int numCol, 
        const int lda);

int genRRTransMat(const float wavNum, const cartCoord* vec, const int numVec, const int p, 
        cuFloatComplex* transMat);

int genSSTransMat(const float wavNum, const cartCoord* vec, const int numVec, const int p, 
        cuFloatComplex* transMat);

int genSRTransMat(const float wavNum, const cartCoord* vec, const int numVec, const int p, 
        cuFloatComplex* transMat);

int genRRCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *mat);

int genSSCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *mat);

int genSRCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *mat);

int genRotMats(const rotAng *rotAngle, const int numRot, const int p, cuFloatComplex *rotMat);

__host__ __device__ void getRotMatBlock(const cuFloatComplex *rotMat, const int p, const int n, 
        cuFloatComplex *rotMatBlock);

__host__ __device__ void getCoaxTransMatBlock(const cuFloatComplex *coaxTransMat, const int p, 
        const int m, cuFloatComplex *coaxTransMatBlock);

__host__ __device__ void cuMatVecMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int len, cuFloatComplex *prod);

__host__ __device__ void cuRotVecMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int p, cuFloatComplex *prod);

__host__ __device__ void cuCoaxTransMatVecMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int p, cuFloatComplex *prod);

__host__ int genRndCoeffs(const int num, cuFloatComplex *coeff);

__host__ int transMatsVecsMul_RR(const float wavNum, const cartCoord *trans, const cuFloatComplex *coeff, 
        const int num, const int p, cuFloatComplex *prod);

__host__ int transMatsVecsMul_RR_rcr(const float wavNum, const cartCoord *trans, 
        const cuFloatComplex *coeff, const int num, const int p, cuFloatComplex *prod);

__host__ int transMatsVecsMul_SS(const float wavNum, const cartCoord *trans, const cuFloatComplex *coeff, 
        const int num, const int p, cuFloatComplex *prod);

__host__ int transMatsVecsMul_SS_rcr(const float wavNum, const cartCoord *trans, 
        const cuFloatComplex *coeff, const int num, const int p, cuFloatComplex *prod);

__host__ int transMatsVecsMul_SR(const float wavNum, const cartCoord *trans, const cuFloatComplex *coeff, 
        const int num, const int p, cuFloatComplex *prod);

__host__ int transMatsVecsMul_SR_rcr(const float wavNum, const cartCoord *trans, 
        const cuFloatComplex *coeff, const int num, const int p, cuFloatComplex *prod);

__host__ __device__ cartCoord triCentroid(const cartCoord nod[3]);
#endif /* NUMERICAL_H */


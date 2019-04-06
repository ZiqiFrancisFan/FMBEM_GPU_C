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

#ifndef abs
#define abs(x) \
({ __typeof__ (x) _x = (x); \
__typeof__ (x) _y = 0; \
_x < _y ? -_x : _x; })
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

__host__ int genRRSparseCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *sparseMat);

int genSSCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *mat);

__host__ int genSSSparseCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *sparseMat);

int genSRCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *mat);

__host__ int genSRSparseCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *sparseMat);

//initialize rotation matrices
__host__ void rotMatsInit(const rotAng *rotAngle, const int numAng, const int p, float *H);

//generate the rotation matrix from the initialized matrix H and the rotation angle rotAngle
__host__ __device__ void rotMatGen(const rotAng rotAngle, const int p, float *H,  
        cuFloatComplex *rotMat);

//generate the sparse rotation matrix from the initialized matrix H and the rotation angle rotAngle
__host__ __device__ void sparseRotMatGen(const rotAng rotAngle, const int p, float *H,  
        cuFloatComplex *sparseRotMat);

__global__ void rotMatsGen(const rotAng *rotAngle, const int numRot, const int p, 
        float *H_enl, cuFloatComplex *rotMat);

__global__ void sparseRotMatsGen(const rotAng *rotAngle, const int numRot, const int p, 
        float *H_enl, cuFloatComplex *sparseRotMat);

__host__ void rrCoaxTransMatsInit(const float wavNum, const float *vec, const int numVec, 
        const int p, cuFloatComplex *mat);

__host__ void ssCoaxTransMatsInit(const float wavNum, const float *vec, const int numVec, 
        const int p, cuFloatComplex *mat);

__host__ void srCoaxTransMatsInit(const float wavNum, const float *vec, const int numVec, 
        const int p, cuFloatComplex *mat);

__host__ __device__ void coaxTransMatGen(cuFloatComplex *enlMat, const int p, cuFloatComplex *mat);

__host__ __device__ void sparseCoaxTransMatGen(cuFloatComplex *enlMat, const int p, cuFloatComplex *sparseMat);

__global__ void sparseCoaxTransMatsGen(cuFloatComplex *enlMat, const int num, const int p, 
        cuFloatComplex *sparseMat);

int genRotMats(const rotAng *rotAngle, const int numRot, const int p, cuFloatComplex *rotMat);

__host__ int genSparseRotMats(const rotAng *rotAngle, const int numRot, const int p, cuFloatComplex *sparseMat);

//get the nth block from the dense rotation matrix of order p
__host__ __device__ void getRotMatBlock(const cuFloatComplex *rotMat, const int p, const int n, 
        cuFloatComplex *rotMatBlock);

//retrieve the nth block rotMatBlock from the sparse matrix rotMat
__host__ __device__ void getBlockFromSparseRotMat(const cuFloatComplex *rotMat, const int n, 
        cuFloatComplex *rotMatBlock);

//convert the rotation matrix from a dense matrix to a sparse matrix
__host__ __device__ void getSparseMatFromRotMat(const cuFloatComplex *rotMat, const int p, 
        cuFloatComplex *sparseRotMat);

//convert an array of rotation matrices from the dense form to the sparse form
__global__ void getSparseMatsFromRotMats(const cuFloatComplex *rotMat, const int num, const int p, 
        cuFloatComplex *sparseRotMat);

//get the nth block from the dense coaxial translation matrix of order p
__host__ __device__ void getCoaxTransMatBlock(const cuFloatComplex *coaxTransMat, const int p, 
        const int m, cuFloatComplex *coaxTransMatBlock);

//retrieve the mth block from the sparse coaxial translation matrix
__host__ __device__ void getBlockFromSparseCoaxTransMat(const cuFloatComplex *coaxTransMat, const int p, 
        const int m, cuFloatComplex *coaxTransMatBlock);

//convert the coaxial translation matrix from a dense matrix to a sparse matrix
__host__ __device__ void getSparseMatFromCoaxTransMat(const cuFloatComplex *coaxTransMat, const int p, 
        cuFloatComplex *sparseCoaxTransMat);

//convert an array of coaxial translation matrices from the dense form to the sparse form
__global__ void getSparseMatsFromCoaxTransMats(const cuFloatComplex *coaxTransMat, const int num, 
        const int p, cuFloatComplex *sparseCoaxTransMat);

__host__ __device__ void cuMatVecMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int len, cuFloatComplex *prod);

__host__ __device__ void cuRotVecMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int p, cuFloatComplex *prod);

__host__ __device__ void cuSparseRotVecMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int p, cuFloatComplex *prod);

__global__ void cuSparseRotsVecsMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int num, const int p, cuFloatComplex *prod);

__host__ __device__ void cuCoaxTransMatVecMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int p, cuFloatComplex *prod);

__host__ __device__ void cuSparseCoaxTransMatVecMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int p, cuFloatComplex *prod);

//generate random coefficients of order num
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

__host__ __device__ void cuMatVecMul_rcr(const cuFloatComplex *rotMat1, const cuFloatComplex *coaxMat, 
        const cuFloatComplex *rotMat2, const cuFloatComplex *vec, const int p, cuFloatComplex *prod);

__host__ int testSparseRotMatsGen(const rotAng *rotAngle, const int numRot, const int p);

__host__ int testSparseCoaxTransMatsGen(const float wavNum, const float *transVec, const int numTransVec, 
        const int p);

//generate SR related rotation angles and coaxial translation vectors of level l
__host__ void genSRCoaxTransVecsRotAngles(const int l, const double d, const cartCoord_d pt_min, 
        float **pVec, int *pNumVec, rotAng **pRotAngle, int *pNumRotAng);

__host__ void genSSCoaxTransVecsRotAngles(const int l, const double d, const cartCoord_d pt_min,
        float **pVec, int *pNumVec, rotAng **pRotAngle, int *pNumRotAng);

__host__ void genRRCoaxTransVecsRotAngles(const int l, const double d, const cartCoord_d pt_min, 
        float **pVec, int *pNumVec, rotAng **pRotAngle, int *pNumRotAng);

//generate indices for rotations and coaxial translations in the SS translation
void genSSIdxArrs(int **fmmLevelSet, const int lmax, const cartCoord_d pt_min, const double d, 
        const float *rrCoaxTransVec, const int numRRTransVec, const rotAng *ang, const int numRot, 
        transIdx **ssTransIdxLevelSet, int **ssTransDestArr);

void genSRIdxArrs(int **fmmLevelSet, int **SRNumLevelArr, const int lmax, const cartCoord_d pt_min, 
        const double d, const float *srCoaxTransVec, const int numSRTransVec, const rotAng *ang, 
        const int numRot, transIdx **srTransIdxLevelArr, int **srTransOriginArr, int **srTransDestArr);

//generate indices for rotations and coaxial translations in the RR translation
void genRRIdxArrs(int **fmmLevelSet, const int lmax, const cartCoord_d pt_min, const double d, 
        const float *rrCoaxTransVec, const int numRRTransVec, const rotAng *ang, const int numRot, 
        transIdx **rrTransIdxLevelSet, int **rrTransOriginArr);

__host__ void initOctree(octree *oct);

__host__ int genOctree(const char *filename, const float wavNum, const int s, octree *oct);

__host__ void destroyOctree(octree *oct);

#endif /* NUMERICAL_H */


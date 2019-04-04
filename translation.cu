/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <time.h>
#include <curand.h>
#include "translation.h"
#include "octree.h"


void printMat_cuFloatComplex(const cuFloatComplex* A, const int numRow, const int numCol, 
        const int lda)
{
    for(int i=0;i<numRow;i++) {
        for(int j=0;j<numCol;j++) {
            printf("(%f,%f)",cuCrealf(A[IDXC0(i,j,lda)]),cuCimagf(A[IDXC0(i,j,lda)]));
        }
        printf("\n");
    }
}

//matrix-vector multiplication; len is the size of the vector and the square matrix
__host__ __device__ void cuMatVecMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int len, cuFloatComplex *prod)
{
    for(int i=0;i<len;i++) {
        prod[i] = make_cuFloatComplex(0,0);
    }
    for(int i=0;i<len;i++) {
        for(int j=0;j<len;j++) {
            prod[i] = cuCaddf(prod[i],cuCmulf(mat[IDXC0(i,j,len)],vec[j]));
        }
    }
}

__global__ void cuMatsVecsMul(const cuFloatComplex *mat, const cuFloatComplex *vec, const int num, 
        const int len, cuFloatComplex *prod)
{
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < num) {
        cuMatVecMul(&mat[len*len*idx],&vec[len*idx],len,&prod[len*idx]);
    }
}

__host__ __device__ void cuMatMatMul(const cuFloatComplex *mat1, const cuFloatComplex *mat2, 
        const int numRow1, const int numCol1, const int numCol2, cuFloatComplex *mat)
{
    for(int i=0;i<numRow1;i++) {
        for(int j=0;j<numCol2;j++) {
            mat[IDXC0(i,j,numRow1)] = make_cuFloatComplex(0,0);
        }
    }
    for(int i=0;i<numRow1;i++) {
        for(int j=0;j<numCol2;j++) {
            for(int k=0;k<numCol1;k++) {
                mat[IDXC0(i,j,numRow1)] = cuCaddf(mat[IDXC0(i,j,numRow1)],
                        cuCmulf(mat1[IDXC0(i,k,numRow1)],mat2[IDXC0(k,j,numCol1)]));
            }
        }
    }
    
}

__host__ __device__ cuFloatComplex cplxExp(const float theta)
{
    return make_cuFloatComplex(cos(theta),sin(theta));
}

__host__ cuFloatComplex gsl_complex2cuFloatComplex(const gsl_complex rhs)
{
    return make_cuFloatComplex(GSL_REAL(rhs),GSL_IMAG(rhs));
}

__host__ gsl_complex gsl_sf_bessel_hl(const int l, const double s)
{
    double x = gsl_sf_bessel_jl(l,s);
    double y = gsl_sf_bessel_yl(l,s);
    gsl_complex z = gsl_complex_rect(x,y);
    return z;
}

__host__ double factorial(const int n)
{
    if(n==0) {
        return 1;
    } else {
        double r = 1; 
        for (int i=2; i<=n; i++) 
            r *= (double)i; 
        return r; 
    }
} 

__host__ double assctdLegendrePly(const int n, const int m, const double u) 
{
    if (m >= 0) {
        return gsl_sf_legendre_Plm(n,m,u);
    } else {
        double a = factorial(n-abs(m));
        double b = factorial(n+abs(m));
        double c = a/b;
        return c*gsl_sf_legendre_Plm(n,abs(m),u);
    }
}

__host__ gsl_complex sphHrmc(const int l, const int m, const double theta, const double phi)
{
    if(abs(m)>l) {
        return gsl_complex_rect(0,0);
    }
    else {
        double x, y;
        double a = factorial(l-abs(m)), b = factorial(l+abs(m));
        double c = a/b;
        x = pow(-1,m)*sqrt((2*l+1)/(4*M_PI)*c)*assctdLegendrePly(l,abs(m),cos(theta))
                *cos(m*phi);
        y = pow(-1,m)*sqrt((2*l+1)/(4*M_PI)*c)*assctdLegendrePly(l,abs(m),cos(theta))
                *sin(m*phi);
        gsl_complex z = gsl_complex_rect(x,y);
        return z;
    }
}

__host__ gsl_complex rglBasis(const double k, const int n, const int m, const sphCoord coord)
{
    double x = gsl_sf_bessel_jl(n,k*coord.r);
    gsl_complex t = sphHrmc(n,m,coord.theta,coord.phi);
    double z_r = x*GSL_REAL(t);
    double z_i = x*GSL_IMAG(t);
    gsl_complex z = gsl_complex_rect(z_r,z_i);
    return z;
}

__host__ gsl_complex sglBasis(const double k, const int n, const int m, const sphCoord coord)
{
    gsl_complex t1 = gsl_sf_bessel_hl(n,k*coord.r);
    gsl_complex t2 = sphHrmc(n,m,coord.theta,coord.phi);
    gsl_complex z = gsl_complex_mul(t1,t2);
    return z;
}

__host__ __device__ float aCoeff(const int n, const int m) {
    float nf = n, mf = m;
    if (nf >= abs(mf)) {
        double result = sqrtf((nf+1+mf)*(nf+1-mf)/((2*nf+1)*(2*nf+3)));
        return result;
    } else {
        return 0;
    }
}

__host__ __device__ float bCoeff(const int n, const int m) {
    float nf = n, mf = m;
    if (mf >= 0 && nf >= mf) {
        float result = sqrtf((nf-mf-1)*(nf-mf)/((2*nf-1)*(2*nf+1)));
        return result;
    } else {
        if (-nf <= mf && mf < 0) {
            float result = -sqrtf((nf-mf-1)*(nf-mf)/((2*nf-1)*(2*nf+1)));
            return result;
        } else {
            return 0;
        }
    }
}

__host__ void rrTransMatsInit(const float wavNum, const cartCoord *vec, const int numVec, 
        const int p, cuFloatComplex *mat)
{
    int np, mp, matRowIdx, matColIdx;
    int matSize = (2*p-1)*(2*p-1); //height or width of the matrix
    cuFloatComplex matElem;
    int matStartIdx, matElemIdx;
    sphCoord coord_sph;
    
    //set all elments to zero
    for(int i=0;i<numVec;i++) {
        matStartIdx = matSize*matSize*i;
        for(matRowIdx=0;matRowIdx<matSize;matRowIdx++) {
            for(matColIdx=0;matColIdx<matSize;matColIdx++) {
                matElemIdx = IDXC0(matRowIdx,matColIdx,matSize);
                mat[matStartIdx+matElemIdx] = make_cuFloatComplex(0,0);
            }
        }
    }
    for(int i=0;i<numVec;i++) {
        //Find the head index of the current translation matrix; 
        coord_sph = cart2sph(vec[i]);
        matStartIdx = matSize*matSize*i;
        for(np=0;np<=2*p-2;np++) {
            for(mp=-np;mp<=np;mp++) {
                matRowIdx = NM2IDX0(np,mp);
                matColIdx = NM2IDX0(0,0);
                matElemIdx = IDXC0(matRowIdx,matColIdx,matSize);
                matElem = gsl_complex2cuFloatComplex(gsl_complex_mul_real(
                        rglBasis(wavNum,np,-mp,coord_sph),sqrt(4*M_PI)*pow(-1,np)));
                mat[matStartIdx+matElemIdx] = matElem;
            }
        }
    }
}

__host__ void ssTransMatsInit(const float wavNum, const cartCoord *vec, const int numVec, 
        const int p, cuFloatComplex *mat)
{
    rrTransMatsInit(wavNum,vec,numVec,p,mat);
}

__host__ void srTransMatsInit(const float wavNum, const cartCoord *vec, const int numVec, 
        const int p, cuFloatComplex *mat)
{
    int np, mp, matRowIdx, matColIdx;
    int matSize = (2*p-1)*(2*p-1); //height or width of the matrix
    cuFloatComplex matElem;
    int matStartIdx, matElemIdx;
    sphCoord coord_sph;
    
    //set all elments to zero
    for(int i=0;i<numVec;i++) {
        matStartIdx = matSize*matSize*i;
        for(matRowIdx=0;matRowIdx<matSize;matRowIdx++) {
            for(matColIdx=0;matColIdx<matSize;matColIdx++) {
                matElemIdx = IDXC0(matRowIdx,matColIdx,matSize);
                mat[matStartIdx+matElemIdx] = make_cuFloatComplex(0,0);
            }
        }
    }
    for(int i=0;i<numVec;i++) {
        //Find the head index of the current translation matrix; 
        coord_sph = cart2sph(vec[i]);
        matStartIdx = matSize*matSize*i;
        for(np=0;np<=2*p-2;np++) {
            for(mp=-np;mp<=np;mp++) {
                matRowIdx = NM2IDX0(np,mp);
                matColIdx = NM2IDX0(0,0);
                matElemIdx = IDXC0(matRowIdx,matColIdx,matSize);
                matElem = gsl_complex2cuFloatComplex(gsl_complex_mul_real(
                        sglBasis(wavNum,np,-mp,coord_sph),sqrt(4*M_PI)*pow(-1,np)));
                mat[matStartIdx+matElemIdx] = matElem;
            }
        }
    }
}

__device__ void transMatGen(cuFloatComplex *matPtr, cuFloatComplex *matPtr2, 
        const int p)
{
    //It is assumed that the matrix is initialized
    int np, mp, n, m, i, j;
    cuFloatComplex temp[6];
    int matIdx = 0;
    const int stride = (2*p-1)*(2*p-1);
    float tp[3];
    
    //The second step in the algorithm
    for(m=0;m+1<=p-1;m++) {
        for(np=0;np<=2*p-2-(m+1);np++) {
            for(mp=-np;mp<=np;mp++) {
                tp[0] = bCoeff(np,-mp)/bCoeff(m+1,-m-1);
                tp[1] = bCoeff(np+1,mp-1)/bCoeff(m+1,-m-1);
                if(np-1>=abs(mp-1)) {
                    i = NM2IDX0(np-1,mp-1);
                    j = NM2IDX0(m,m);
                    matIdx = IDXC0(i,j,stride);
                    temp[0] = matPtr[matIdx];
                } else {
                    temp[0] = make_cuFloatComplex(0,0);
                }
                i = NM2IDX0(np+1,mp-1);
                j = NM2IDX0(m,m);
                matIdx = IDXC0(i,j,stride);
                temp[1] = matPtr[matIdx];
                temp[0] = make_cuFloatComplex(tp[0]*cuCrealf(temp[0]),
                        tp[0]*cuCimagf(temp[0]));
                temp[1] = make_cuFloatComplex(tp[1]*cuCrealf(temp[1]),
                        tp[1]*cuCimagf(temp[1]));
                temp[2] = cuCsubf(temp[0],temp[1]);
                i = NM2IDX0(np,mp);
                j = NM2IDX0(m+1,m+1);
                matIdx = IDXC0(i,j,stride);
                matPtr[matIdx] = temp[2];
            }
        }
    }
    
    //The third step in the algorithm
    for(m=0;-m-1>=-(p-1);m++) {
        for(np=0;np<=2*p-2-(m+1);np++) {
            for(mp=-np;mp<=np;mp++) {
                tp[0] = bCoeff(np,mp)/bCoeff(m+1,-m-1);
                tp[1] = bCoeff(np+1,-mp-1)/bCoeff(m+1,-m-1);
                if(np-1>=abs(mp+1)) {
                    i = NM2IDX0(np-1,mp+1);
                    j = NM2IDX0(-m,m);
                    matIdx = IDXC0(i,j,stride);
                    temp[0] = matPtr[matIdx];
                    
                } else {
                    temp[0] = make_cuFloatComplex(0,0);
                }
                i = NM2IDX0(np+1,mp+1);
                j = NM2IDX0(-m,m);
                matIdx = IDXC0(i,j,stride);
                temp[1] = matPtr[matIdx];
                temp[0] = make_cuFloatComplex(tp[0]*cuCrealf(temp[0]),
                        tp[0]*cuCimagf(temp[0]));
                temp[1] = make_cuFloatComplex(tp[1]*cuCrealf(temp[1]),
                        tp[1]*cuCimagf(temp[1]));
                temp[2] = cuCsubf(temp[0],temp[1]);
                i = NM2IDX0(np,mp);
                j = NM2IDX0(-m-1,m+1);
                matIdx = IDXC0(i,j,stride);
                matPtr[matIdx] = temp[2];
            }
        }
    }
    //printf("Completed 3rd step: \n");
    //printMat_cuFltCplx(matPtr,stride,stride,stride);
    //The fourth step in the algorithm
    for(mp=-(p-1);mp<=p-1;mp++) {
        for(n=0;n<=2*p-2-abs(mp);n++) {
            for(m=-n;m<=n;m++) {
                i = NM2IDX0(n,-m);
                j = NM2IDX0(abs(mp),-mp);
                matIdx = IDXC0(i,j,stride);
                temp[0] = matPtr[matIdx];
                tp[0] = pow(-1,n+abs(mp));
                temp[1] = make_cuFloatComplex(tp[0]*cuCrealf(temp[0]),
                        tp[0]*cuCimagf(temp[0]));
                i = NM2IDX0(abs(mp),mp);
                
                j = NM2IDX0(n,m);
                
                matIdx = IDXC0(i,j,stride);
                
                matPtr[matIdx] = temp[1];
            }
        }
    }
    
    for(m=-(p-2);m<=p-2;m++) {
        for(n=abs(m);n+1<=p-1;n++) {
            for(np=1;np<=2*p-2-(n+1);np++) {
                for(mp=-np;mp<=np;mp++) {
                    tp[0] = aCoeff(n-1,m)/aCoeff(n,m);
                    tp[1] = aCoeff(np-1,mp)/aCoeff(n,m);
                    tp[2] = aCoeff(np,mp)/aCoeff(n,m);
                    if(n-1>=abs(m)) {
                        i = NM2IDX0(np,mp);
                        j = NM2IDX0(n-1,m);
                        matIdx = IDXC0(i,j,stride);
                        temp[0] = matPtr[matIdx];
                    } else {
                        temp[0] = make_cuFloatComplex(0,0);
                    }
                    if(np-1>=abs(mp)) {
                        i = NM2IDX0(np-1,mp);
                        j = NM2IDX0(n,m);
                        matIdx = IDXC0(i,j,stride);
                        temp[1] = matPtr[matIdx];
                    } else {
                        temp[1] = make_cuFloatComplex(0,0);
                    }
                    i = NM2IDX0(np+1,mp);
                    j = NM2IDX0(n,m);
                    matIdx = IDXC0(i,j,stride);
                    temp[2] = matPtr[matIdx];
                    temp[0] = make_cuFloatComplex(tp[0]*cuCrealf(temp[0]),
                            tp[0]*cuCimagf(temp[0]));
                    temp[1] = make_cuFloatComplex(tp[1]*cuCrealf(temp[1]),
                            tp[1]*cuCimagf(temp[1]));
                    temp[2] = make_cuFloatComplex(tp[2]*cuCrealf(temp[2]),
                            tp[2]*cuCimagf(temp[2]));
                    i = NM2IDX0(np,mp);
                    j = NM2IDX0(n+1,m);
                    matIdx = IDXC0(i,j,stride);
                    matPtr[matIdx] = cuCsubf(cuCaddf(temp[0],temp[1]),temp[2]);
                }
            }
        }
    }
    
    for(mp=-(p-2);mp<=(p-2);mp++) {
        for(np=abs(mp);np+1<=p-1;np++) {
            for(n=1;n<=2*p-2-(np+1);n++) {
                for(m=-n;m<=n;m++) {
                    tp[0] = aCoeff(n-1,m)/aCoeff(np,mp);
                    tp[1] = aCoeff(np-1,mp)/aCoeff(np,mp);
                    tp[2] = aCoeff(n,m)/aCoeff(np,mp);
                    if(n-1>=abs(m)) {
                        i = NM2IDX0(np,mp);
                        j = NM2IDX0(n-1,m);
                        matIdx = IDXC0(i,j,stride);
                        temp[0] = matPtr[matIdx];
                    } else {
                        temp[0] = make_cuFloatComplex(0,0);
                    }
                    if(np-1>=abs(mp)) {
                        i = NM2IDX0(np-1,mp);
                        j = NM2IDX0(n,m);
                        matIdx = IDXC0(i,j,stride);
                        temp[1] = matPtr[matIdx];
                    } else {
                        temp[1] = make_cuFloatComplex(0,0);
                    }
                    i = NM2IDX0(np,mp);
                    j = NM2IDX0(n+1,m);
                    matIdx = IDXC0(i,j,stride);
                    temp[2] = matPtr[matIdx];
                    temp[0] = make_cuFloatComplex(tp[0]*cuCrealf(temp[0]),
                            tp[0]*cuCimagf(temp[0]));
                    temp[1] = make_cuFloatComplex(tp[1]*cuCrealf(temp[1]),
                            tp[1]*cuCimagf(temp[1]));
                    temp[2] = make_cuFloatComplex(tp[2]*cuCrealf(temp[2]),
                            tp[2]*cuCimagf(temp[2]));
                    i = NM2IDX0(np+1,mp);
                    j = NM2IDX0(n,m);
                    matIdx = IDXC0(i,j,stride);
                    //printf("np+1=%d,mp=%d,n=%d,m=%d,i=%d,j=%d\n",np+1,mp,n,m,i,j);
                    matPtr[matIdx] = cuCsubf(cuCaddf(temp[0],temp[1]),temp[2]);
                }
            }
        }
    }
    for(i=0;i<p*p;i++) {
        for(j=0;j<p*p;j++) {
            matIdx = IDXC0(i,j,stride);
            //printf("matIdx: %d\n",matIdx);
            temp[0] = matPtr[matIdx];
            matIdx = IDXC0(i,j,p*p);
            matPtr2[matIdx] = temp[0];
        }
    }
}

__global__ void transMatsGen(cuFloatComplex *mats_enl,const int maxNum, 
        const int p, cuFloatComplex *mats)
{
    int idx_x = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx_x < maxNum) {
        transMatGen(&mats_enl[idx_x*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)],&mats[idx_x*p*p*p*p],p);
    }
}

int genRRTransMat(const float wavNum, const cartCoord* vec, const int numVec, const int p, 
        cuFloatComplex* transMat)
{
    int matSize;
    matSize = (2*p-1)*(2*p-1);
    cuFloatComplex *enlMat = (cuFloatComplex*)malloc(numVec*matSize*matSize*sizeof(cuFloatComplex));
    rrTransMatsInit(wavNum,vec,numVec,p,enlMat); //Initalization completed
    cuFloatComplex *enlMat_d;
    CUDA_CALL(cudaMalloc(&enlMat_d,numVec*matSize*matSize*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(enlMat_d,enlMat,numVec*matSize*matSize*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    cuFloatComplex *transMat_d;
    CUDA_CALL(cudaMalloc(&transMat_d,numVec*p*p*p*p*sizeof(cuFloatComplex)));
    
    int numBlocksPerGrid, numThreadsPerBlock = 32;
    numBlocksPerGrid = (numVec+numThreadsPerBlock-1)/numThreadsPerBlock;
    
    dim3 gridStruct(numBlocksPerGrid,1,1);
    dim3 blockStruct(numThreadsPerBlock,1,1);
    transMatsGen<<<gridStruct,blockStruct>>>(enlMat_d,numVec,p,transMat_d);
    
    CUDA_CALL(cudaMemcpy(transMat,transMat_d,numVec*p*p*p*p*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(transMat_d));
    CUDA_CALL(cudaFree(enlMat_d));
    free(enlMat);
    return EXIT_SUCCESS;
}

int genSSTransMat(const float wavNum, const cartCoord* vec, const int numVec, const int p, 
        cuFloatComplex* transMat)
{
    HOST_CALL(genRRTransMat(wavNum,vec,numVec,p,transMat));
    return EXIT_SUCCESS;
}

int genSRTransMat(const float wavNum, const cartCoord* vec, const int numVec, const int p, 
        cuFloatComplex* transMat)
{
    int matSize;
    matSize = (2*p-1)*(2*p-1);
    cuFloatComplex *enlMat = (cuFloatComplex*)malloc(numVec*matSize*matSize*sizeof(cuFloatComplex));
    srTransMatsInit(wavNum,vec,numVec,p,enlMat); //Initalization completed
    cuFloatComplex *enlMat_d;
    CUDA_CALL(cudaMalloc(&enlMat_d,numVec*matSize*matSize*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(enlMat_d,enlMat,numVec*matSize*matSize*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    cuFloatComplex *transMat_d;
    CUDA_CALL(cudaMalloc(&transMat_d,numVec*p*p*p*p*sizeof(cuFloatComplex)));
    
    int numBlocksPerGrid, numThreadsPerBlock = 32;
    numBlocksPerGrid = (numVec+numThreadsPerBlock-1)/numThreadsPerBlock;
    
    dim3 gridStruct(numBlocksPerGrid,1,1);
    dim3 blockStruct(numThreadsPerBlock,1,1);
    transMatsGen<<<gridStruct,blockStruct>>>(enlMat_d,numVec,p,transMat_d);
    
    CUDA_CALL(cudaMemcpy(transMat,transMat_d,numVec*p*p*p*p*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(transMat_d));
    CUDA_CALL(cudaFree(enlMat_d));
    free(enlMat);
    return EXIT_SUCCESS;
}

__host__ void rrCoaxTransMatsInit(const float wavNum, const float *vec, const int numVec, 
        const int p, cuFloatComplex *mat)
{
    int np, matRowIdx, matColIdx;
    int matSize = (2*p-1)*(2*p-1);
    cuFloatComplex matElem;
    int matStartIdx, matElemIdx, i;
    
    for(int i=0;i<numVec;i++) {
        matStartIdx = matSize*matSize*i;
        for(matRowIdx=0;matRowIdx<matSize;matRowIdx++) {
            for(matColIdx=0;matColIdx<matSize;matColIdx++) {
                matElemIdx = IDXC0(matRowIdx,matColIdx,matSize);
                mat[matStartIdx+matElemIdx] = make_cuFloatComplex(0,0);
            }
        }
    }
    
    for(i=0;i<numVec;i++) {
        //Find the head index of the current translation matrix; 
        matStartIdx = matSize*matSize*i;
        for(np=0;np<=2*p-2;np++) {
            matRowIdx = NM2IDX0(np,0);
            matColIdx = NM2IDX0(0,0);
            matElemIdx = IDXC0(matRowIdx,matColIdx,matSize);
            matElem = make_cuFloatComplex(pow(-1,np)*sqrt(2*np+1)*gsl_sf_bessel_jl(np,wavNum*vec[i]),0);
            mat[matStartIdx+matElemIdx] = matElem;
        }
    }
}

__host__ __device__ void coaxTransMatGen(cuFloatComplex *enlMat, const int p, cuFloatComplex *mat)
{
    int np, n, m;
    int i, j;
    cuFloatComplex temp[6];
    float tp[3];
    int idx;
    
    for(m=0;m+1<=p-1;m++) {
        for(np=m+1;np<=2*p-2-(m+1);np++) {
            tp[0] = bCoeff(np,-m-1)/bCoeff(m+1,-m-1);
            tp[1] = bCoeff(np+1,m)/bCoeff(m+1,-m-1);
            i = NM2IDX0(np-1,m);
            j = NM2IDX0(m,m);
            idx = IDXC0(i,j,(2*p-1)*(2*p-1));
            temp[0] = enlMat[idx];
            i = NM2IDX0(np+1,m);
            j = NM2IDX0(m,m);
            idx = IDXC0(i,j,(2*p-1)*(2*p-1));
            temp[1] = enlMat[idx];
            temp[2] = make_cuFloatComplex(tp[0]*cuCrealf(temp[0]),tp[0]*cuCimagf(temp[0]));
            temp[3] = make_cuFloatComplex(tp[1]*cuCrealf(temp[1]),tp[1]*cuCimagf(temp[1]));
            temp[4] = cuCsubf(temp[2],temp[3]);
            i = NM2IDX0(np,m+1);
            j = NM2IDX0(m+1,m+1);
            idx = IDXC0(i,j,(2*p-1)*(2*p-1));
            enlMat[idx] = temp[4];
        }
    }
    
    for(m=0;m<=p-2;m++) {
        for(n=m;n+1<=p-1;n++) {
            for(np=m;np<=2*p-2-(n+1);np++) {
                tp[0] = aCoeff(np-1,m)/aCoeff(n,m);
                tp[1] = aCoeff(np,m)/aCoeff(n,m);
                tp[2] = aCoeff(n-1,m)/aCoeff(n,m);
                if(np-1>=m) {
                    i = NM2IDX0(np-1,m);
                    j = NM2IDX0(n,m);
                    idx = IDXC0(i,j,(2*p-1)*(2*p-1));
                    temp[0] = enlMat[idx];
                } else {
                    temp[0] = make_cuFloatComplex(0,0);
                }
                i = NM2IDX0(np+1,m);
                j = NM2IDX0(n,m);
                idx = IDXC0(i,j,(2*p-1)*(2*p-1));
                temp[1] = enlMat[idx];
                if(n-1>=m) {
                    i = NM2IDX0(np,m);
                    j = NM2IDX0(n-1,m);
                    idx = IDXC0(i,j,(2*p-1)*(2*p-1));
                    temp[2] = enlMat[idx];
                } else {
                    temp[2] = make_cuFloatComplex(0,0);
                }
                temp[3] = make_cuFloatComplex(tp[0]*cuCrealf(temp[0]),tp[0]*cuCimagf(temp[0]));
                temp[4] = make_cuFloatComplex(tp[1]*cuCrealf(temp[1]),tp[1]*cuCimagf(temp[1]));
                temp[5] = make_cuFloatComplex(tp[2]*cuCrealf(temp[2]),tp[2]*cuCimagf(temp[2]));
                i = NM2IDX0(np,m);
                j = NM2IDX0(n+1,m);
                idx = IDXC0(i,j,(2*p-1)*(2*p-1));
                enlMat[idx] = cuCaddf(cuCsubf(temp[3],temp[4]),temp[5]);
            }
        }
    }
    
    for(m=-1;m>=-(p-1);m--) {
        for(np=abs(m);np<=p-1;np++) {
            for(n=abs(m);n<=p-1;n++) {
                i = NM2IDX0(np,abs(m));
                j = NM2IDX0(n,abs(m));
                idx = IDXC0(i,j,(2*p-1)*(2*p-1));
                temp[0] = enlMat[idx];
                i = NM2IDX0(np,m);
                j = NM2IDX0(n,m);
                idx = IDXC0(i,j,(2*p-1)*(2*p-1));
                enlMat[idx] = temp[0];
            }
        }
    }
    
    for(i=0;i<=p*p-1;i++) {
        for(j=0;j<=p*p-1;j++) {
            idx = IDXC0(i,j,(2*p-1)*(2*p-1));
            temp[0] = enlMat[idx];
            idx = IDXC0(i,j,p*p);
            mat[idx] = temp[0];
        }
    }
}

__host__ __device__ void sparseCoaxTransMatGen(cuFloatComplex *enlMat, const int p, cuFloatComplex *sparseMat)
{
    int np, n, m, idx_s;
    int i, j;
    cuFloatComplex temp[6];
    float tp[3];
    int idx;
    
    for(m=0;m+1<=p-1;m++) {
        for(np=m+1;np<=2*p-2-(m+1);np++) {
            tp[0] = bCoeff(np,-m-1)/bCoeff(m+1,-m-1);
            tp[1] = bCoeff(np+1,m)/bCoeff(m+1,-m-1);
            i = NM2IDX0(np-1,m);
            j = NM2IDX0(m,m);
            idx = IDXC0(i,j,(2*p-1)*(2*p-1));
            temp[0] = enlMat[idx];
            i = NM2IDX0(np+1,m);
            j = NM2IDX0(m,m);
            idx = IDXC0(i,j,(2*p-1)*(2*p-1));
            temp[1] = enlMat[idx];
            temp[2] = make_cuFloatComplex(tp[0]*cuCrealf(temp[0]),tp[0]*cuCimagf(temp[0]));
            temp[3] = make_cuFloatComplex(tp[1]*cuCrealf(temp[1]),tp[1]*cuCimagf(temp[1]));
            temp[4] = cuCsubf(temp[2],temp[3]);
            i = NM2IDX0(np,m+1);
            j = NM2IDX0(m+1,m+1);
            idx = IDXC0(i,j,(2*p-1)*(2*p-1));
            enlMat[idx] = temp[4];
        }
    }
    
    for(m=0;m<=p-2;m++) {
        for(n=m;n+1<=p-1;n++) {
            for(np=m;np<=2*p-2-(n+1);np++) {
                tp[0] = aCoeff(np-1,m)/aCoeff(n,m);
                tp[1] = aCoeff(np,m)/aCoeff(n,m);
                tp[2] = aCoeff(n-1,m)/aCoeff(n,m);
                if(np-1>=m) {
                    i = NM2IDX0(np-1,m);
                    j = NM2IDX0(n,m);
                    idx = IDXC0(i,j,(2*p-1)*(2*p-1));
                    temp[0] = enlMat[idx];
                } else {
                    temp[0] = make_cuFloatComplex(0,0);
                }
                i = NM2IDX0(np+1,m);
                j = NM2IDX0(n,m);
                idx = IDXC0(i,j,(2*p-1)*(2*p-1));
                temp[1] = enlMat[idx];
                if(n-1>=m) {
                    i = NM2IDX0(np,m);
                    j = NM2IDX0(n-1,m);
                    idx = IDXC0(i,j,(2*p-1)*(2*p-1));
                    temp[2] = enlMat[idx];
                } else {
                    temp[2] = make_cuFloatComplex(0,0);
                }
                temp[3] = make_cuFloatComplex(tp[0]*cuCrealf(temp[0]),tp[0]*cuCimagf(temp[0]));
                temp[4] = make_cuFloatComplex(tp[1]*cuCrealf(temp[1]),tp[1]*cuCimagf(temp[1]));
                temp[5] = make_cuFloatComplex(tp[2]*cuCrealf(temp[2]),tp[2]*cuCimagf(temp[2]));
                i = NM2IDX0(np,m);
                j = NM2IDX0(n+1,m);
                idx = IDXC0(i,j,(2*p-1)*(2*p-1));
                enlMat[idx] = cuCaddf(cuCsubf(temp[3],temp[4]),temp[5]);
            }
        }
    }
    
    for(m=-1;m>=-(p-1);m--) {
        for(np=abs(m);np<=p-1;np++) {
            for(n=abs(m);n<=p-1;n++) {
                i = NM2IDX0(np,abs(m));
                j = NM2IDX0(n,abs(m));
                idx = IDXC0(i,j,(2*p-1)*(2*p-1));
                temp[0] = enlMat[idx];
                i = NM2IDX0(np,m);
                j = NM2IDX0(n,m);
                idx = IDXC0(i,j,(2*p-1)*(2*p-1));
                enlMat[idx] = temp[0];
            }
        }
    }
    
    for(m=0;m<p;m++) {
        idx_s = m*(2*m*m-(3+6*p)*m+1+6*(p+p*p))/6;
        for(np=m;np<p;np++) {
            i = NM2IDX0(np,m);
            for(n=m;n<p;n++) {
                j = NM2IDX0(n,m);
                temp[0] = enlMat[IDXC0(i,j,(2*p-1)*(2*p-1))];
                sparseMat[idx_s+IDXC0(np-m,n-m,p-m)] = temp[0];
            }
        }
    }
}

__global__ void coaxTransMatsGen(cuFloatComplex *mats_enl, const int maxNum, const int p, 
        cuFloatComplex *mats)
{
    int idx_x = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx_x < maxNum) {
        coaxTransMatGen(&mats_enl[idx_x*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)],p,&mats[idx_x*p*p*p*p]);
    }
}

__global__ void sparseCoaxTransMatsGen(cuFloatComplex *enlMat, const int num, const int p, 
        cuFloatComplex *sparseMat)
{
    int idx_x = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx_x < num) {
        sparseCoaxTransMatGen(&enlMat[idx_x*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)],p,&sparseMat[idx_x*(p*(2*p*p+3*p+1)/6)]);
    }
}

int genRRCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *mat)
{
    cuFloatComplex *enlMat_h = (cuFloatComplex*)malloc(
            numVec*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex));
    rrCoaxTransMatsInit(wavNum,vec,numVec,p,enlMat_h);
    
    int numBlocksPerGrid, numThreadsPerBlock = 32;
    numBlocksPerGrid = (numVec+numThreadsPerBlock-1)/numThreadsPerBlock;
    
    dim3 gridStruct(numBlocksPerGrid,1,1);
    dim3 blockStruct(numThreadsPerBlock,1,1);
    
    cuFloatComplex *enlMat_d, *mat_d;
    CUDA_CALL(cudaMalloc(&enlMat_d,numVec*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMalloc(&mat_d,numVec*p*p*p*p*sizeof(cuFloatComplex)));
    
    
    CUDA_CALL(cudaMemcpy(enlMat_d,enlMat_h,numVec*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex)
            ,cudaMemcpyHostToDevice));
    
    coaxTransMatsGen<<<gridStruct,blockStruct>>>(enlMat_d,numVec,p,mat_d);
    
    CUDA_CALL(cudaMemcpy(mat,mat_d,sizeof(cuFloatComplex)*p*p*p*p*numVec,cudaMemcpyDeviceToHost));
    
    CUDA_CALL(cudaFree(enlMat_d));
    CUDA_CALL(cudaFree(mat_d));
    free(enlMat_h);
    return EXIT_SUCCESS;
}

__host__ int genRRSparseCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *sparseMat)
{
    //compute the number of matrices that can be allocated each time on the GPU
    size_t matMem = ((2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)+p*(2*p*p+3*p+1)/6)*sizeof(cuFloatComplex), 
            totalMem, freeMem;
    CUDA_CALL(cudaMemGetInfo(&freeMem,&totalMem));
    int numMatsPerAlloc = floor((float)freeMem*0.95/matMem), numGen = (numVec+numMatsPerAlloc-1)/numMatsPerAlloc;
    
    int restNumVec; //the remaining number of vectors not processed
    cuFloatComplex *enlMat_h, *enlMat_d, *sparseMat_d; //pointers for host and device memory
    
    //iterate through all blocks
    for(int i=0;i<numGen;i++) {
        //update the rest number of vectors to be processed
        restNumVec = numVec-i*numMatsPerAlloc;
        
        //tell if the rest number of vectors is smaller than numMatsPerAlloc
        if(restNumVec>=numMatsPerAlloc) {
            //Allocate memory and initialize the matrices
            enlMat_h = (cuFloatComplex*)malloc(numMatsPerAlloc*(2*p-1)*(2*p-1)
                    *(2*p-1)*(2*p-1)*sizeof(cuFloatComplex));
            
            rrCoaxTransMatsInit(wavNum,&vec[i*numMatsPerAlloc],numMatsPerAlloc,p,enlMat_h);
            
            //compute the number of blocks per grid
            int numBlocksPerGrid, numThreadsPerBlock = 32;
            numBlocksPerGrid = (numMatsPerAlloc+numThreadsPerBlock-1)/numThreadsPerBlock;
            dim3 gridStruct(numBlocksPerGrid,1,1);
            dim3 blockStruct(numThreadsPerBlock,1,1);
            
            //allocate memory of the enlarged matrices, dense matrices and sparse matrices on the device
            CUDA_CALL(cudaMalloc(&enlMat_d,numMatsPerAlloc*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex)));
            CUDA_CALL(cudaMalloc(&sparseMat_d,numMatsPerAlloc*(p*(2*p*p+3*p+1)/6)*sizeof(cuFloatComplex)));
            
            //copy initialized enlarged matrices from host to device
            CUDA_CALL(cudaMemcpy(enlMat_d,enlMat_h,numMatsPerAlloc*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex)
                    ,cudaMemcpyHostToDevice));
            free(enlMat_h);
            
            //update coaxial translation matrices and copy them to the host
            sparseCoaxTransMatsGen<<<gridStruct,blockStruct>>>(enlMat_d,numMatsPerAlloc,p,sparseMat_d);
            CUDA_CALL(cudaFree(enlMat_d));
            
            CUDA_CALL(cudaMemcpy(&sparseMat[i*numMatsPerAlloc*(p*(2*p*p+3*p+1)/6)],sparseMat_d,
                    sizeof(cuFloatComplex)*(p*(2*p*p+3*p+1)/6)*numMatsPerAlloc,cudaMemcpyDeviceToHost));
        } else {
            //allocate memory and initialize the matrices
            enlMat_h = (cuFloatComplex*)malloc(restNumVec*(2*p-1)*(2*p-1)
                    *(2*p-1)*(2*p-1)*sizeof(cuFloatComplex));
            
            rrCoaxTransMatsInit(wavNum,&vec[i*numMatsPerAlloc],restNumVec,p,enlMat_h);
            
            //compute the number of blocks per grid
            int numBlocksPerGrid, numThreadsPerBlock = 32;
            numBlocksPerGrid = (restNumVec+numThreadsPerBlock-1)/numThreadsPerBlock;
            dim3 gridStruct(numBlocksPerGrid,1,1);
            dim3 blockStruct(numThreadsPerBlock,1,1);
            
            //allocate memory of the enlarged matrices, dense matrices and sparse matrices on the device
            CUDA_CALL(cudaMalloc(&enlMat_d,restNumVec*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex)));
            CUDA_CALL(cudaMalloc(&sparseMat_d,restNumVec*(p*(2*p*p+3*p+1)/6)*sizeof(cuFloatComplex)));
            
            //copy initialized enlarged matrices from host to device
            CUDA_CALL(cudaMemcpy(enlMat_d,enlMat_h,restNumVec*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex)
                    ,cudaMemcpyHostToDevice));
            free(enlMat_h);
            
            //update coaxial translation matrices and copy them to the host
            sparseCoaxTransMatsGen<<<gridStruct,blockStruct>>>(enlMat_d,restNumVec,p,sparseMat_d);
            CUDA_CALL(cudaFree(enlMat_d));
            
            CUDA_CALL(cudaMemcpy(&sparseMat[i*numMatsPerAlloc*(p*(2*p*p+3*p+1)/6)],sparseMat_d,
                    sizeof(cuFloatComplex)*(p*(2*p*p+3*p+1)/6)*restNumVec,cudaMemcpyDeviceToHost));
        }
        
        CUDA_CALL(cudaFree(sparseMat_d));
        
    }
    
    return EXIT_SUCCESS;
}

int genSSCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *mat)
{
    HOST_CALL(genRRCoaxTransMat(wavNum,vec,numVec,p,mat));
    return EXIT_SUCCESS;
}

__host__ int genSSSparseCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *sparseMat)
{
    HOST_CALL(genRRSparseCoaxTransMat(wavNum,vec,numVec,p,sparseMat));
    return EXIT_SUCCESS;
}

__host__ void srCoaxTransMatsInit(const float wavNum, const float *vec, const int numVec, 
        const int p, cuFloatComplex *mat)
{
    int np, matRowIdx, matColIdx;
    int matSize = (2*p-1)*(2*p-1);
    cuFloatComplex matElem;
    int matStartIdx, matElemIdx, i;
    
    for(int i=0;i<numVec;i++) {
        matStartIdx = matSize*matSize*i;
        for(matRowIdx=0;matRowIdx<matSize;matRowIdx++) {
            for(matColIdx=0;matColIdx<matSize;matColIdx++) {
                matElemIdx = IDXC0(matRowIdx,matColIdx,matSize);
                mat[matStartIdx+matElemIdx] = make_cuFloatComplex(0,0);
            }
        }
    }
    for(i=0;i<numVec;i++) {
        //Find the head index of the current translation matrix; 
        matStartIdx = matSize*matSize*i;
        for(np=0;np<=2*p-2;np++) {
            matRowIdx = NM2IDX0(np,0);
            matColIdx = NM2IDX0(0,0);
            matElemIdx = IDXC0(matRowIdx,matColIdx,matSize);
            matElem = gsl_complex2cuFloatComplex(gsl_sf_bessel_hl(np,wavNum*vec[i]));
            matElem = make_cuFloatComplex(powf(-1,np)*sqrtf(2*np+1)*cuCrealf(matElem),
                    powf(-1,np)*sqrtf(2*np+1)*cuCimagf(matElem));
            mat[matStartIdx+matElemIdx] = matElem;
        }
    }
}

int genSRCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *mat)
{
    cuFloatComplex *enlMat_h = (cuFloatComplex*)malloc(
            numVec*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex));
    srCoaxTransMatsInit(wavNum,vec,numVec,p,enlMat_h);
    
    int numBlocksPerGrid, numThreadsPerBlock = 32;
    numBlocksPerGrid = (numVec+numThreadsPerBlock-1)/numThreadsPerBlock;
    
    dim3 gridStruct(numBlocksPerGrid,1,1);
    dim3 blockStruct(numThreadsPerBlock,1,1);
    
    cuFloatComplex *enlMat_d, *mat_d;
    CUDA_CALL(cudaMalloc(&enlMat_d,numVec*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMalloc(&mat_d,numVec*p*p*p*p*sizeof(cuFloatComplex)));
    
    
    CUDA_CALL(cudaMemcpy(enlMat_d,enlMat_h,numVec*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex)
            ,cudaMemcpyHostToDevice));
    
    coaxTransMatsGen<<<gridStruct,blockStruct>>>(enlMat_d,numVec,p,mat_d);
    
    CUDA_CALL(cudaMemcpy(mat,mat_d,sizeof(cuFloatComplex)*p*p*p*p*numVec,cudaMemcpyDeviceToHost));
    
    CUDA_CALL(cudaFree(enlMat_d));
    CUDA_CALL(cudaFree(mat_d));
    free(enlMat_h);
    return EXIT_SUCCESS;
}

__host__ int genSRSparseCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *sparseMat)
{
    //compute the number of matrices that can be allocated each time on the GPU
    size_t matMem = ((2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)+p*(2*p*p+3*p+1)/6)*sizeof(cuFloatComplex), 
            totalMem, freeMem;
    CUDA_CALL(cudaMemGetInfo(&freeMem,&totalMem));
    int numMatsPerAlloc = floor((float)freeMem*0.95/matMem), numGen = (numVec+numMatsPerAlloc-1)/numMatsPerAlloc;
    int restNumVec; //the remaining number of vectors not processed
    cuFloatComplex *enlMat_h, *enlMat_d, *sparseMat_d; //pointers for host and device memory
    
    //iterate through all blocks
    for(int i=0;i<numGen;i++) {
        //update the rest number of vectors to be processed
        restNumVec = numVec-i*numMatsPerAlloc;
        
        //tell if the rest number of vectors is smaller than numMatsPerAlloc
        if(restNumVec>=numMatsPerAlloc) {
            //Allocate memory and initialize the matrices
            enlMat_h = (cuFloatComplex*)malloc(numMatsPerAlloc*(2*p-1)*(2*p-1)
                    *(2*p-1)*(2*p-1)*sizeof(cuFloatComplex));
            
            srCoaxTransMatsInit(wavNum,&vec[i*numMatsPerAlloc],numMatsPerAlloc,p,enlMat_h);
            
            //compute the number of blocks per grid
            int numBlocksPerGrid, numThreadsPerBlock = 32;
            numBlocksPerGrid = (numMatsPerAlloc+numThreadsPerBlock-1)/numThreadsPerBlock;
            dim3 gridStruct(numBlocksPerGrid,1,1);
            dim3 blockStruct(numThreadsPerBlock,1,1);
            
            //allocate memory of the enlarged matrices, dense matrices and sparse matrices on the device
            CUDA_CALL(cudaMalloc(&enlMat_d,numMatsPerAlloc*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex)));
            CUDA_CALL(cudaMalloc(&sparseMat_d,numMatsPerAlloc*(p*(2*p*p+3*p+1)/6)*sizeof(cuFloatComplex)));
            
            //copy initialized enlarged matrices from host to device
            CUDA_CALL(cudaMemcpy(enlMat_d,enlMat_h,numMatsPerAlloc*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex)
                    ,cudaMemcpyHostToDevice));
            free(enlMat_h);
            
            //update coaxial translation matrices and copy them to the host
            sparseCoaxTransMatsGen<<<gridStruct,blockStruct>>>(enlMat_d,numMatsPerAlloc,p,sparseMat_d);
            CUDA_CALL(cudaFree(enlMat_d));
            
            CUDA_CALL(cudaMemcpy(&sparseMat[i*numMatsPerAlloc*(p*(2*p*p+3*p+1)/6)],sparseMat_d,
                    sizeof(cuFloatComplex)*(p*(2*p*p+3*p+1)/6)*numMatsPerAlloc,cudaMemcpyDeviceToHost));
        } else {
            //allocate memory and initialize the matrices
            enlMat_h = (cuFloatComplex*)malloc(restNumVec*(2*p-1)*(2*p-1)
                    *(2*p-1)*(2*p-1)*sizeof(cuFloatComplex));
            
            srCoaxTransMatsInit(wavNum,&vec[i*numMatsPerAlloc],restNumVec,p,enlMat_h);
            
            //compute the number of blocks per grid
            int numBlocksPerGrid, numThreadsPerBlock = 32;
            numBlocksPerGrid = (restNumVec+numThreadsPerBlock-1)/numThreadsPerBlock;
            dim3 gridStruct(numBlocksPerGrid,1,1);
            dim3 blockStruct(numThreadsPerBlock,1,1);
            
            //allocate memory of the enlarged matrices, dense matrices and sparse matrices on the device
            CUDA_CALL(cudaMalloc(&enlMat_d,restNumVec*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex)));
            CUDA_CALL(cudaMalloc(&sparseMat_d,restNumVec*(p*(2*p*p+3*p+1)/6)*sizeof(cuFloatComplex)));
            
            //copy initialized enlarged matrices from host to device
            CUDA_CALL(cudaMemcpy(enlMat_d,enlMat_h,restNumVec*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(cuFloatComplex)
                    ,cudaMemcpyHostToDevice));
            free(enlMat_h);
            
            //update coaxial translation matrices and copy them to the host
            sparseCoaxTransMatsGen<<<gridStruct,blockStruct>>>(enlMat_d,restNumVec,p,sparseMat_d);
            CUDA_CALL(cudaFree(enlMat_d));
            
            CUDA_CALL(cudaMemcpy(&sparseMat[i*numMatsPerAlloc*(p*(2*p*p+3*p+1)/6)],sparseMat_d,
                    sizeof(cuFloatComplex)*(p*(2*p*p+3*p+1)/6)*restNumVec,cudaMemcpyDeviceToHost));
        }
        
        CUDA_CALL(cudaFree(sparseMat_d));
        
    }
    
    return EXIT_SUCCESS;
}

__host__ void rotMatsInit(const rotAng *rotAngle, const int numAng, const int p, float *H)
{
    float temp;
    int n, mp, i, j;
    int matSize = (2*p-1)*(2*p-1);
    int matStartIdx, matElemIdx, vecIdx, matRowIdx, matColIdx;
    for(int i=0;i<numAng;i++) {
        matStartIdx = matSize*matSize*i;
        for(matRowIdx=0;matRowIdx<matSize;matRowIdx++) {
            for(matColIdx=0;matColIdx<matSize;matColIdx++) {
                matElemIdx = IDXC0(matRowIdx,matColIdx,matSize);
                H[matStartIdx+matElemIdx] = 0;
            }
        }
    }
    for(vecIdx=0;vecIdx<numAng;vecIdx++) {
        for(n=0;n<=2*p-2;n++) {
            for(mp=-n;mp<=n;mp++) {
                i = NM2IDX0(n,mp);
                j = NM2IDX0(n,0);
                matElemIdx = IDXC0(i,j,matSize);
                matStartIdx = vecIdx*matSize*matSize;
                temp = pow(-1,mp)*sqrt(factorial(n-abs(mp))/factorial(n+abs(mp)))
                        *assctdLegendrePly(n,abs(mp),cos(rotAngle[vecIdx].beta));
                H[matStartIdx+matElemIdx] = temp;
            }
        }
    }
}

__host__ __device__ void rotMatGen(const rotAng rotAngle, const int p, float *H,  
        cuFloatComplex *rotMat)
{
    int np, n, mp, m, i, j, matIdx;
    float temp[6];
    cuFloatComplex z;
    for(m=0;m+1<=p-1;m++) {
        for(n=m+1;n<=2*p-2-(m+1);n++) {
            for(mp=-n;mp<=n;mp++) {
                //printf("n=%d,mp=%d,m=%d\n",n,mp,m);
                temp[0] = 0.5*bCoeff(n+1,-mp-1)/bCoeff(n+1,m)*(1-cos(rotAngle.beta));
                temp[1] = 0.5*bCoeff(n+1,mp-1)/bCoeff(n+1,m)*(1+cos(rotAngle.beta));
                temp[2] = aCoeff(n,mp)/bCoeff(n+1,m)*sin(rotAngle.beta);
                i = NM2IDX0(n+1,mp+1);
                j = NM2IDX0(n+1,m);
                matIdx = IDXC0(i,j,(2*p-1)*(2*p-1));
                temp[3] = H[matIdx];
                i = NM2IDX0(n+1,mp-1);
                j = NM2IDX0(n+1,m);
                matIdx = IDXC0(i,j,(2*p-1)*(2*p-1));
                temp[4] = H[matIdx];
                i = NM2IDX0(n+1,mp);
                j = NM2IDX0(n+1,m);
                matIdx = IDXC0(i,j,(2*p-1)*(2*p-1));
                temp[5] = H[matIdx];
                i = NM2IDX0(n,mp);
                j = NM2IDX0(n,m+1);
                matIdx = IDXC0(i,j,(2*p-1)*(2*p-1));
                H[matIdx] = temp[0]*temp[3]-temp[1]*temp[4]-temp[2]*temp[5];
            }
        }
    }
    
    for(m=-1;m>=-(p-1);m--) {
        for(n=abs(m);n<=p-1;n++) {
            for(mp=-n;mp<=n;mp++) {
                i = NM2IDX0(n,-mp);
                j = NM2IDX0(n,-m);
                matIdx = IDXC0(i,j,(2*p-1)*(2*p-1));
                temp[0] = H[matIdx];
                i = NM2IDX0(n,mp);
                j = NM2IDX0(n,m);
                matIdx = IDXC0(i,j,(2*p-1)*(2*p-1));
                H[matIdx] = temp[0];
            }
        }
    }
    
    for(np=0;np<=p-1;np++) {
        for(mp=-np;mp<=np;mp++) {
            i = NM2IDX0(np,mp);
            for(n=0;n<=p-1;n++) {
                for(m=-n;m<=n;m++) {
                    j = NM2IDX0(n,m);
                    matIdx = IDXC0(i,j,p*p);
                    rotMat[matIdx] = make_cuFloatComplex(0,0);
                }
            }
        }
    }
    
    for(n=0;n<=p-1;n++) {
        for(mp=-n;mp<=n;mp++) {
            i = NM2IDX0(n,mp);
            for(m=-n;m<=n;m++) {
                j = NM2IDX0(n,m);
                z = cuCmulf(cplxExp(m*rotAngle.alpha),cplxExp(-mp*rotAngle.gamma));
                matIdx = IDXC0(i,j,(2*p-1)*(2*p-1));
                temp[0] = H[matIdx];
                z = make_cuFloatComplex(cuCrealf(z)*temp[0],cuCimagf(z)*temp[0]);
                matIdx = IDXC0(i,j,p*p);
                rotMat[matIdx] = z;
            }
        }
    }
}

__host__ __device__ void sparseRotMatGen(const rotAng rotAngle, const int p, float *H,  
        cuFloatComplex *sparseRotMat)
{
    int n, mp, m, i, j, matIdx, idx_s;
    float temp[6];
    cuFloatComplex z;
    
    //Process the matrix H
    for(m=0;m+1<=p-1;m++) {
        for(n=m+1;n<=2*p-2-(m+1);n++) {
            for(mp=-n;mp<=n;mp++) {
                //printf("n=%d,mp=%d,m=%d\n",n,mp,m);
                temp[0] = 0.5*bCoeff(n+1,-mp-1)/bCoeff(n+1,m)*(1-cos(rotAngle.beta));
                temp[1] = 0.5*bCoeff(n+1,mp-1)/bCoeff(n+1,m)*(1+cos(rotAngle.beta));
                temp[2] = aCoeff(n,mp)/bCoeff(n+1,m)*sin(rotAngle.beta);
                i = NM2IDX0(n+1,mp+1);
                j = NM2IDX0(n+1,m);
                matIdx = IDXC0(i,j,(2*p-1)*(2*p-1));
                temp[3] = H[matIdx];
                i = NM2IDX0(n+1,mp-1);
                j = NM2IDX0(n+1,m);
                matIdx = IDXC0(i,j,(2*p-1)*(2*p-1));
                temp[4] = H[matIdx];
                i = NM2IDX0(n+1,mp);
                j = NM2IDX0(n+1,m);
                matIdx = IDXC0(i,j,(2*p-1)*(2*p-1));
                temp[5] = H[matIdx];
                i = NM2IDX0(n,mp);
                j = NM2IDX0(n,m+1);
                matIdx = IDXC0(i,j,(2*p-1)*(2*p-1));
                H[matIdx] = temp[0]*temp[3]-temp[1]*temp[4]-temp[2]*temp[5];
            }
        }
    }
    
    for(m=-1;m>=-(p-1);m--) {
        for(n=abs(m);n<=p-1;n++) {
            for(mp=-n;mp<=n;mp++) {
                i = NM2IDX0(n,-mp);
                j = NM2IDX0(n,-m);
                matIdx = IDXC0(i,j,(2*p-1)*(2*p-1));
                temp[0] = H[matIdx];
                i = NM2IDX0(n,mp);
                j = NM2IDX0(n,m);
                matIdx = IDXC0(i,j,(2*p-1)*(2*p-1));
                H[matIdx] = temp[0];
            }
        }
    }
    
    for(n=0;n<p;n++) {
        idx_s = n*(4*n*n-1)/3; //starting index in the sparse matrix
        for(mp=-n;mp<=n;mp++) {
            i = NM2IDX0(n,mp);
            for(m=-n;m<=n;m++) {
                j = NM2IDX0(n,m);
                matIdx = IDXC0(i,j,(2*p-1)*(2*p-1));
                temp[0] = H[matIdx];
                z = cuCmulf(cplxExp(m*rotAngle.alpha),cplxExp(-mp*rotAngle.gamma));
                z = make_cuFloatComplex(cuCrealf(z)*temp[0],cuCimagf(z)*temp[0]);
                matIdx = IDXC0(mp+n,m+n,2*n+1); //index in the sub-matrix of the sparse matrix
                sparseRotMat[idx_s+matIdx] = z;
            }
        }
    }
    
}

__global__ void rotMatsGen(const rotAng *rotAngle, const int numRot, const int p, 
        float *H_enl, cuFloatComplex *rotMat)
{
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < numRot) {
        rotMatGen(rotAngle[idx],p,&H_enl[idx*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)],&rotMat[idx*p*p*p*p]);
    }
}

__global__ void sparseRotMatsGen(const rotAng *rotAngle, const int numRot, const int p, 
        float *H_enl, cuFloatComplex *sparseRotMat)
{
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < numRot) {
        sparseRotMatGen(rotAngle[idx],p,&H_enl[idx*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)],&sparseRotMat[idx*(p*(4*p*p-1)/3)]);
    }
}

int genRotMats(const rotAng *rotAngle, const int numRot, const int p, cuFloatComplex *rotMat)
{
    float *H = (float*)malloc(numRot*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(float));
    
    //initialize the Hs_enl;
    rotMatsInit(rotAngle,numRot,p,H);
    
    //Allocate memory for rotation angles
    
    //Transfer rotation angles to the GPU
    rotAng *rotAngle_d;
    CUDA_CALL(cudaMalloc(&rotAngle_d,numRot*sizeof(rotAng)));
    CUDA_CALL(cudaMemcpy(rotAngle_d,rotAngle,numRot*sizeof(rotAng),cudaMemcpyHostToDevice));
    
    //Transfer the matrix H to global memory
    float *H_d;
    CUDA_CALL(cudaMalloc(&H_d,numRot*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(float)));
    CUDA_CALL(cudaMemcpy(H_d,H,numRot*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(float),cudaMemcpyHostToDevice));
    
    //Allocate memory for rotation matrices
    cuFloatComplex *rotMat_d;
    CUDA_CALL(cudaMalloc(&rotMat_d,numRot*p*p*p*p*sizeof(cuFloatComplex)));
    
    
    int numBlocksPerGrid, numThreadsPerBlock = 32;
    numBlocksPerGrid = (numRot+numThreadsPerBlock-1)/numThreadsPerBlock;
    
    dim3 gridStruct(numBlocksPerGrid,1,1);
    dim3 blockStruct(numThreadsPerBlock,1,1);
    
    rotMatsGen<<<gridStruct,blockStruct>>>(rotAngle_d,numRot,p,H_d,rotMat_d);
    CUDA_CALL(cudaMemcpy(rotMat,rotMat_d,numRot*p*p*p*p*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
   
    CUDA_CALL(cudaFree(H_d));
    CUDA_CALL(cudaFree(rotMat_d));
    CUDA_CALL(cudaFree(rotAngle_d));
    free(H);
    return EXIT_SUCCESS;
}

__host__ int genSparseRotMats(const rotAng *rotAngle, const int numRot, const int p, cuFloatComplex *sparseMat)
{
    //compute the memory required for each sparse rotation matrix
    size_t matMem = (2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(float)+(p*(4*p*p-1)/3+(2*p-1)*(2*p-1))*sizeof(cuFloatComplex)
            +sizeof(rotAng), totalMem, freeMem;
    
    //get gpu memory information
    CUDA_CALL(cudaMemGetInfo(&freeMem,&totalMem));
    
    //compute the number of generations
    int numMatsPerAlloc = floor((float)freeMem*0.95/matMem), numGen = (numRot+numMatsPerAlloc-1)/numMatsPerAlloc;
    int restNumRot; //the remaining number of rotations to be processed
    
    //allocate memory
    rotAng *rotAngle_d;
    float *enlMat_h, *enlMat_d;
    cuFloatComplex *sparseMat_d; //pointers for host and device memory
    
    //iterate through the generations
    for(int i=0;i<numGen;i++) {
        //update the rest number of vectors to be processed
        restNumRot = numRot-i*numMatsPerAlloc;
        
        //tell if the rest number of vectors is smaller than numMatsPerAlloc
        if(restNumRot>=numMatsPerAlloc) {
            //allocate memory for rotation angles and copy the angles from host to device
            CUDA_CALL(cudaMalloc(&rotAngle_d,numMatsPerAlloc*sizeof(rotAng)));
            CUDA_CALL(cudaMemcpy(rotAngle_d,&rotAngle[i*numMatsPerAlloc],numMatsPerAlloc*sizeof(rotAng),
                    cudaMemcpyHostToDevice));
            
            //allocate memory and initialize the matrices
            enlMat_h = (float*)malloc(numMatsPerAlloc*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(float));
            rotMatsInit(&rotAngle[i*numMatsPerAlloc],numMatsPerAlloc,p,enlMat_h);
            
            //compute the number of blocks per grid
            int numBlocksPerGrid, numThreadsPerBlock = 32;
            numBlocksPerGrid = (numMatsPerAlloc+numThreadsPerBlock-1)/numThreadsPerBlock;
            dim3 gridStruct(numBlocksPerGrid,1,1);
            dim3 blockStruct(numThreadsPerBlock,1,1);
            
            //allocate memory of the enlarged matrices, dense matrices and sparse matrices on the device
            CUDA_CALL(cudaMalloc(&enlMat_d,numMatsPerAlloc*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(float)));
            CUDA_CALL(cudaMalloc(&sparseMat_d,numMatsPerAlloc*(p*(4*p*p-1)/3)*sizeof(cuFloatComplex)));
            
            //copy initialized enlarged matrices from host to device
            CUDA_CALL(cudaMemcpy(enlMat_d,enlMat_h,numMatsPerAlloc*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(float)
                    ,cudaMemcpyHostToDevice));
            free(enlMat_h);
            
            //update rotation matrices and copy them to the host
            sparseRotMatsGen<<<gridStruct,blockStruct>>>(rotAngle_d,numMatsPerAlloc,p,enlMat_d,sparseMat_d);
            CUDA_CALL(cudaFree(rotAngle_d));
            CUDA_CALL(cudaFree(enlMat_d));
            
            CUDA_CALL(cudaMemcpy(&sparseMat[i*numMatsPerAlloc*(p*(4*p*p-1)/3)],sparseMat_d,
                    sizeof(cuFloatComplex)*(p*(4*p*p-1)/3)*numMatsPerAlloc,cudaMemcpyDeviceToHost));
        } else {
            //allocate memory for rotation angles and copy the angles from host to device
            CUDA_CALL(cudaMalloc(&rotAngle_d,restNumRot*sizeof(rotAng)));
            CUDA_CALL(cudaMemcpy(rotAngle_d,&rotAngle[i*numMatsPerAlloc],restNumRot*sizeof(rotAng),
                    cudaMemcpyHostToDevice));
            
            //allocate memory and initialize the matrices
            enlMat_h = (float*)malloc(restNumRot*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(float));
            
            rotMatsInit(&rotAngle[i*numMatsPerAlloc],restNumRot,p,enlMat_h);
            
            //compute the number of blocks per grid
            int numBlocksPerGrid, numThreadsPerBlock = 32;
            numBlocksPerGrid = (restNumRot+numThreadsPerBlock-1)/numThreadsPerBlock;
            dim3 gridStruct(numBlocksPerGrid,1,1);
            dim3 blockStruct(numThreadsPerBlock,1,1);
            
            //allocate memory of the enlarged matrices, dense matrices and sparse matrices on the device
            CUDA_CALL(cudaMalloc(&enlMat_d,restNumRot*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(float)));
            CUDA_CALL(cudaMalloc(&sparseMat_d,restNumRot*(p*(4*p*p-1)/3)*sizeof(cuFloatComplex)));
            
            //copy initialized enlarged matrices from host to device
            CUDA_CALL(cudaMemcpy(enlMat_d,enlMat_h,restNumRot*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*sizeof(float)
                    ,cudaMemcpyHostToDevice));
            free(enlMat_h);
            
            //update rotation matrices and copy them to the host
            sparseRotMatsGen<<<gridStruct,blockStruct>>>(rotAngle_d,restNumRot,p,enlMat_d,sparseMat_d);
            CUDA_CALL(cudaFree(rotAngle_d));
            CUDA_CALL(cudaFree(enlMat_d));
            
            CUDA_CALL(cudaMemcpy(&sparseMat[i*numMatsPerAlloc*(p*(4*p*p-1)/3)],sparseMat_d,
                    sizeof(cuFloatComplex)*(p*(4*p*p-1)/3)*restNumRot,cudaMemcpyDeviceToHost));
        }
        
        //release memory for dense and sparse matrices
        CUDA_CALL(cudaFree(sparseMat_d));
        
    }
    
    return EXIT_SUCCESS;
}

__host__ __device__ void getRotMatBlock(const cuFloatComplex *rotMat, const int p, const int n, 
        cuFloatComplex *rotMatBlock)
{
    int mp, m;
    for(mp=-n;mp<=n;mp++) {
        for(m=-n;m<=n;m++) {
            rotMatBlock[IDXC0(mp+n,m+n,2*n+1)] = rotMat[IDXC0(NM2IDX0(n,mp),NM2IDX0(n,m),p*p)];
        }
    }
}

//retrieve the nth block rotMatBlock from the matrix rotMat
__host__ __device__ void getBlockFromSparseRotMat(const cuFloatComplex *rotMat, const int n, 
        cuFloatComplex *rotMatBlock)
{
    int idx_s = n*(4*n*n-1)/3;
    //int idx_e = (n+1)*(4*(n+1)*(n+1)-1)/3-1;
    for(int i=0;i<(2*n+1)*(2*n+1);i++) {
        rotMatBlock[i] = rotMat[idx_s+i];
    }
}

//convert the rotation matrix from a dense matrix to a sparse matrix
__host__ __device__ void getSparseMatFromRotMat(const cuFloatComplex *rotMat, const int p, 
        cuFloatComplex *sparseRotMat)
{
    int idx, matSize;
    cuFloatComplex *matBlock;
    for(int n=0;n<p;n++) {
        matSize = 2*n+1;
        matBlock = (cuFloatComplex*)malloc(matSize*matSize*sizeof(cuFloatComplex));
        getRotMatBlock(rotMat,p,n,matBlock);
        idx = n*(4*n*n-1)/3;
        for(int i=0;i<matSize*matSize;i++) {
            sparseRotMat[idx+i] = matBlock[i];
        }
        free(matBlock);
    }
}

//convert an array of rotation matrices from the dense form to the sparse form
__global__ void getSparseMatsFromRotMats(const cuFloatComplex *rotMat, const int num, const int p, 
        cuFloatComplex *sparseRotMat)
{
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < num) {
        int denseMatSize = p*p*p*p, sparseMatSize = p*(4*p*p-1)/3;
        getSparseMatFromRotMat(&rotMat[idx*denseMatSize],p,&sparseRotMat[idx*sparseMatSize]);
    }
}

__host__ __device__ void getCoaxTransMatBlock(const cuFloatComplex *coaxTransMat, const int p, 
        const int m, cuFloatComplex *coaxTransMatBlock)
{
    int np, n;
    int matsize = p-abs(m);
    for(np=abs(m);np<=p-1;np++) {
        for(n=abs(m);n<=p-1;n++) {
            coaxTransMatBlock[IDXC0(np-abs(m),n-abs(m),matsize)] 
                    = coaxTransMat[IDXC0(NM2IDX0(np,m),NM2IDX0(n,m),p*p)];
        }
    }
}

__host__ __device__ void getBlockFromSparseCoaxTransMat(const cuFloatComplex *coaxTransMat, const int p, 
        const int m, cuFloatComplex *coaxTransMatBlock)
{
    int idx_s = abs(m)*(2*abs(m)*abs(m)-(3+6*p)*abs(m)+1+6*(p+p*p))/6;
    int matsize = p-abs(m);
    for(int i=0;i<matsize*matsize;i++) {
        coaxTransMatBlock[i] = coaxTransMat[idx_s+i];
    }
}

__host__ __device__ void getSparseMatFromCoaxTransMat(const cuFloatComplex *coaxTransMat, const int p, 
        cuFloatComplex *sparseCoaxTransMat)
{
    int idx, matSize;
    cuFloatComplex *matBlock;
    for(int m=0;m<p;m++) {
        idx = m*(2*m*m-(3+6*p)*m+1+6*(p+p*p))/6;
        matSize = (p-m)*(p-m);
        matBlock = (cuFloatComplex*)malloc(matSize*sizeof(cuFloatComplex));
        getCoaxTransMatBlock(coaxTransMat,p,m,matBlock);
        for(int i=0;i<matSize;i++) {
            sparseCoaxTransMat[idx+i] = matBlock[i];
        }
        free(matBlock);
    }
}

__global__ void getSparseMatsFromCoaxTransMats(const cuFloatComplex *coaxTransMat, const int num, 
        const int p, cuFloatComplex *sparseCoaxTransMat)
{
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < num) {
        int denseMatSize = p*p*p*p, sparseMatSize = p*(2*p*p+3*p+1)/6;
        getSparseMatFromCoaxTransMat(&coaxTransMat[idx*denseMatSize],p,&sparseCoaxTransMat[idx*sparseMatSize]);
    }
}

__host__ __device__ void getRotVecBlock(const cuFloatComplex *vec, const int n, cuFloatComplex *v)
{
    int m;
    for(m=-n;m<=n;m++) {
        v[m+n] = vec[NM2IDX0(n,m)];
    }
}

__host__ __device__ void getCoaxVecBlock(const cuFloatComplex *vec, const int p, const int m, 
        cuFloatComplex *v)
{
    int n;
    for(n=abs(m);n<p;n++) {
        v[n-abs(m)] = vec[NM2IDX0(n,m)];
    }
}

__host__ __device__ void sendRotVecBlock(const cuFloatComplex *v, const int n, cuFloatComplex *vec)
{
    int m;
    for(m=-n;m<=n;m++) {
        vec[NM2IDX0(n,m)] = v[m+n];
    }
}

__host__ __device__ void sendCoaxVecBlock(const cuFloatComplex *v, const int p, const int m, 
        cuFloatComplex *vec)
{
    int n;
    for(n=abs(m);n<p;n++) {
        vec[NM2IDX0(n,m)] = v[n-abs(m)];
    }
}

__host__ __device__ void cuRotVecMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int p, cuFloatComplex *prod)
{
    int n, matsize;
    cuFloatComplex *blockMat, *blockVec, *blockProd;
    for(n=0;n<p;n++) {
        matsize = 2*n+1;
        blockMat = (cuFloatComplex*)malloc(matsize*matsize*sizeof(cuFloatComplex));
        blockVec = (cuFloatComplex*)malloc(matsize*sizeof(cuFloatComplex));
        blockProd = (cuFloatComplex*)malloc(matsize*sizeof(cuFloatComplex));
        getRotMatBlock(mat,p,n,blockMat);
        getRotVecBlock(vec,n,blockVec);
        cuMatVecMul(blockMat,blockVec,matsize,blockProd);
        sendRotVecBlock(blockProd,n,prod);
        free(blockMat);
        free(blockVec);
        free(blockProd);
    }
}

//sparse rotation matrix and vector multiplication; mat is of size p*(4*p*p-1)/3, vec is of size p*p
__host__ __device__ void cuSparseRotVecMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int p, cuFloatComplex *prod)
{
    int matsize;
    cuFloatComplex *blockMat, *blockVec, *blockProd;
    for(int n=0;n<p;n++) {
        matsize = 2*n+1;
        blockMat = (cuFloatComplex*)malloc(matsize*matsize*sizeof(cuFloatComplex));
        blockVec = (cuFloatComplex*)malloc(matsize*sizeof(cuFloatComplex));
        blockProd = (cuFloatComplex*)malloc(matsize*sizeof(cuFloatComplex));
        //getRotMatBlock(mat,p,n,blockMat);
        getBlockFromSparseRotMat(mat,n,blockMat);
        getRotVecBlock(vec,n,blockVec);
        cuMatVecMul(blockMat,blockVec,matsize,blockProd);
        sendRotVecBlock(blockProd,n,prod);
        free(blockMat);
        free(blockVec);
        free(blockProd);
    }
}

__host__ __device__ void cuCoaxTransMatVecMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int p, cuFloatComplex *prod)
{
    int m, matsize;
    cuFloatComplex *blockMat, *blockVec, *blockProd;
    for(m=-(p-1);m<=p-1;m++) {
        matsize = p-abs(m);
        blockMat = (cuFloatComplex*)malloc(matsize*matsize*sizeof(cuFloatComplex));
        blockVec = (cuFloatComplex*)malloc(matsize*sizeof(cuFloatComplex));
        blockProd = (cuFloatComplex*)malloc(matsize*sizeof(cuFloatComplex));
        getCoaxTransMatBlock(mat,p,m,blockMat);
        getCoaxVecBlock(vec,p,m,blockVec);
        cuMatVecMul(blockMat,blockVec,matsize,blockProd);
        sendCoaxVecBlock(blockProd,p,m,prod);
        free(blockProd);
        free(blockVec);
        free(blockMat);
    }
}

__host__ __device__ void cuSparseCoaxTransMatVecMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int p, cuFloatComplex *prod)
{
    int matSize;
    cuFloatComplex *blockMat, *blockVec, *blockProd;
    for(int m=-(p-1);m<=p-1;m++) {
        matSize = p-abs(m);
        blockMat = (cuFloatComplex*)malloc(matSize*matSize*sizeof(cuFloatComplex));
        blockVec = (cuFloatComplex*)malloc(matSize*sizeof(cuFloatComplex));
        blockProd = (cuFloatComplex*)malloc(matSize*sizeof(cuFloatComplex));
        getBlockFromSparseCoaxTransMat(mat,p,m,blockMat);
        getCoaxVecBlock(vec,p,m,blockVec);
        cuMatVecMul(blockMat,blockVec,matSize,blockProd);
        sendCoaxVecBlock(blockProd,p,m,prod);
        free(blockProd);
        free(blockVec);
        free(blockMat);
    }
}

__global__ void cuRotsVecsMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int num, const int p, cuFloatComplex *prod)
{
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < num) {
        int len = p*p;
        cuRotVecMul(&mat[idx*len*len],&vec[idx*len],p,&prod[idx*len]);
    }
}

__global__ void cuSparseRotsVecsMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int num, const int p, cuFloatComplex *prod)
{
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < num) {
        int matSize = p*(4*p*p-1)/3;
        int vecSize = p*p;
        cuSparseRotVecMul(&mat[idx*matSize],&vec[idx*vecSize],p,&prod[idx*vecSize]);
    }
}

__global__ void cuCoaxTransMatsVecsMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int num, const int p, cuFloatComplex *prod)
{
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < num) {
        int len = p*p;
        cuCoaxTransMatVecMul(&mat[idx*len*len],&vec[idx*len],p,&prod[idx*len]);
    }
}

__global__ void cuSparseCoaxTransMatsVecsMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int num, const int p, cuFloatComplex *prod)
{
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < num) {
        int matSize = p*(2*p*p+3*p+1)/6;
        int vecSize = p*p;
        cuSparseCoaxTransMatVecMul(&mat[idx*matSize],&vec[idx*vecSize],p,&prod[idx*vecSize]);
    }
}

__host__ int genRndCoeffs(const int num, cuFloatComplex *coeff)
{
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGeneratorHost(&gen,CURAND_RNG_PSEUDO_DEFAULT)); //construct generator
    unsigned long long seed = time(NULL);
    float rndNum[2];
    for(int i=0;i<num;i++) {
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,seed++));
        CURAND_CALL(curandGenerateUniform(gen,rndNum,2));
        coeff[i] = make_cuFloatComplex(rndNum[0],rndNum[1]);
    }
    CURAND_CALL(curandDestroyGenerator(gen));
    
    return EXIT_SUCCESS;
}

__host__ int transMatsVecsMul_RR(const float wavNum, const cartCoord *trans, const cuFloatComplex *coeff, 
        const int num, const int p, cuFloatComplex *prod)
{
    int numBlocksPerGrid, numThreadsPerBlock = 32;
    numBlocksPerGrid = (num+numThreadsPerBlock-1)/numThreadsPerBlock;
    
    dim3 gridStruct(numBlocksPerGrid,1,1);
    dim3 blockStruct(numThreadsPerBlock,1,1);
    
    cuFloatComplex *enlMat_h = (cuFloatComplex*)malloc((2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*num*sizeof(cuFloatComplex));
    rrTransMatsInit(wavNum,trans,num,p,enlMat_h);
    
    cuFloatComplex *enlMat_d;
    CUDA_CALL(cudaMalloc(&enlMat_d,(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*num*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(enlMat_d,enlMat_h,(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*num*sizeof(cuFloatComplex),
            cudaMemcpyHostToDevice));
    
    cuFloatComplex *mat_d;
    CUDA_CALL(cudaMalloc(&mat_d,p*p*p*p*num*sizeof(cuFloatComplex)));
    transMatsGen<<<gridStruct,blockStruct>>>(enlMat_d,num,p,mat_d);
    
    cuFloatComplex *coeff_d;
    CUDA_CALL(cudaMalloc(&coeff_d,p*p*num*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(coeff_d,coeff,p*p*num*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    cuFloatComplex *prod_d;
    CUDA_CALL(cudaMalloc(&prod_d,p*p*num*sizeof(cuFloatComplex)));
    cuMatsVecsMul<<<gridStruct,blockStruct>>>(mat_d,coeff_d,num,p*p,prod_d);
    
    CUDA_CALL(cudaMemcpy(prod,prod_d,p*p*num*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    
    CUDA_CALL(cudaFree(prod_d));
    CUDA_CALL(cudaFree(coeff_d));
    CUDA_CALL(cudaFree(mat_d));
    CUDA_CALL(cudaFree(enlMat_d));
    free(enlMat_h);
    return EXIT_SUCCESS;
}

__host__ int transMatsVecsMul_RR_rcr(const float wavNum, const cartCoord *trans, const cuFloatComplex *coeff, 
        const int num, const int p, cuFloatComplex *prod)
{
    //Rotation angles for the first rotation operator
    rotAng *rotAngle_h = (rotAng*)malloc(num*sizeof(rotAng));
    float *t_h = (float*)malloc(num*sizeof(float));
    sphCoord sphTemp;
    float temp;
    
    for(int i=0;i<num;i++) {
        sphTemp = cart2sph(trans[i]);
        rotAngle_h[i].alpha = sphTemp.phi;
        rotAngle_h[i].beta = sphTemp.theta;
        rotAngle_h[i].gamma = 0;
        t_h[i] = sphTemp.r;
    }
    
    rotAng *rotAngle_d;
    CUDA_CALL(cudaMalloc(&rotAngle_d,num*sizeof(rotAng)));
    CUDA_CALL(cudaMemcpy(rotAngle_d,rotAngle_h,num*sizeof(rotAng),cudaMemcpyHostToDevice));
    
    //Set up kernel launch parameters
    int numBlocksPerGrid, numThreadsPerBlock = 32;
    numBlocksPerGrid = (num+numThreadsPerBlock-1)/numThreadsPerBlock;
    
    dim3 gridStruct(numBlocksPerGrid,1,1);
    dim3 blockStruct(numThreadsPerBlock,1,1);
    
    //First rotation operator
    float *H_h = (float*)malloc((2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*num*sizeof(float));
    rotMatsInit(rotAngle_h,num,p,H_h);
    
    float *H_d;
    CUDA_CALL(cudaMalloc(&H_d,(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*num*sizeof(float)));
    CUDA_CALL(cudaMemcpy(H_d,H_h,(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*num*sizeof(float),cudaMemcpyHostToDevice));
    
    cuFloatComplex *rotMat_d;
    CUDA_CALL(cudaMalloc(&rotMat_d,p*p*p*p*num*sizeof(cuFloatComplex)));
    rotMatsGen<<<gridStruct,blockStruct>>>(rotAngle_d,num,p,H_d,rotMat_d);
    
    cuFloatComplex *coeff_d;
    CUDA_CALL(cudaMalloc(&coeff_d,p*p*num*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(coeff_d,coeff,p*p*num*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    cuFloatComplex *prod_d;
    CUDA_CALL(cudaMalloc(&prod_d,p*p*num*sizeof(cuFloatComplex)));
    cuRotsVecsMul<<<gridStruct,blockStruct>>>(rotMat_d,coeff_d,num,p,prod_d);
    CUDA_CALL(cudaMemcpy(coeff_d,prod_d,p*p*num*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
    
    //Coaxial translation operator
    float *t_d;
    CUDA_CALL(cudaMalloc(&t_d,num*sizeof(float)));
    CUDA_CALL(cudaMemcpy(t_d,t_h,num*sizeof(float),cudaMemcpyHostToDevice));
    
    cuFloatComplex *enlCoaxTransMat_h = (cuFloatComplex*)malloc((2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*
            num*sizeof(cuFloatComplex));
    rrCoaxTransMatsInit(wavNum,t_h,num,p,enlCoaxTransMat_h);
    cuFloatComplex *enlCoaxTransMat_d;
    CUDA_CALL(cudaMalloc(&enlCoaxTransMat_d,(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*
            num*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(enlCoaxTransMat_d,enlCoaxTransMat_h,(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*
            num*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    cuFloatComplex *coaxTransMat_d;
    CUDA_CALL(cudaMalloc(&coaxTransMat_d,p*p*p*p*num*sizeof(cuFloatComplex)));
    coaxTransMatsGen<<<gridStruct,blockStruct>>>(enlCoaxTransMat_d,num,p,coaxTransMat_d);
    
    cuCoaxTransMatsVecsMul<<<gridStruct,blockStruct>>>(coaxTransMat_d,coeff_d,num,p,prod_d);
    CUDA_CALL(cudaMemcpy(coeff_d,prod_d,p*p*num*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
    
    //Second rotation operator
    for(int i=0;i<num;i++) {
        temp = rotAngle_h[i].alpha;
        rotAngle_h[i].alpha = rotAngle_h[i].gamma;
        rotAngle_h[i].gamma = temp;
    }
    rotMatsInit(rotAngle_h,num,p,H_h);
    CUDA_CALL(cudaMemcpy(H_d,H_h,(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*num*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(rotAngle_d,rotAngle_h,num*sizeof(rotAng),cudaMemcpyHostToDevice));
    
    rotMatsGen<<<gridStruct,blockStruct>>>(rotAngle_d,num,p,H_d,rotMat_d);
    cuRotsVecsMul<<<gridStruct,blockStruct>>>(rotMat_d,coeff_d,num,p,prod_d);
    CUDA_CALL(cudaMemcpy(prod,prod_d,p*p*num*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    CUDA_CALL(cudaFree(prod_d));
    CUDA_CALL(cudaFree(coeff_d));
    CUDA_CALL(cudaFree(enlCoaxTransMat_d));
    CUDA_CALL(cudaFree(coaxTransMat_d));
    CUDA_CALL(cudaFree(H_d));
    CUDA_CALL(cudaFree(rotMat_d));
    CUDA_CALL(cudaFree(rotAngle_d));
    
    free(H_h);
    free(enlCoaxTransMat_h);
    free(t_h);
    free(rotAngle_h);
    return EXIT_SUCCESS;
}

__host__ int transMatsVecsMul_SS(const float wavNum, const cartCoord *trans, const cuFloatComplex *coeff, 
        const int num, const int p, cuFloatComplex *prod)
{
    HOST_CALL(transMatsVecsMul_RR(wavNum,trans,coeff,num,p,prod));
    return EXIT_SUCCESS;
}

__host__ int transMatsVecsMul_SS_rcr(const float wavNum, const cartCoord *trans, const cuFloatComplex *coeff, 
        const int num, const int p, cuFloatComplex *prod)
{
    HOST_CALL(transMatsVecsMul_RR_rcr(wavNum,trans,coeff,num,p,prod));
    return EXIT_SUCCESS;
}

__host__ int transMatsVecsMul_SR(const float wavNum, const cartCoord *trans, const cuFloatComplex *coeff, 
        const int num, const int p, cuFloatComplex *prod)
{
    int numBlocksPerGrid, numThreadsPerBlock = 32;
    numBlocksPerGrid = (num+numThreadsPerBlock-1)/numThreadsPerBlock;
    
    dim3 gridStruct(numBlocksPerGrid,1,1);
    dim3 blockStruct(numThreadsPerBlock,1,1);
    
    cuFloatComplex *enlMat_h = (cuFloatComplex*)malloc((2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*num*sizeof(cuFloatComplex));
    srTransMatsInit(wavNum,trans,num,p,enlMat_h);
    
    cuFloatComplex *enlMat_d;
    CUDA_CALL(cudaMalloc(&enlMat_d,(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*num*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(enlMat_d,enlMat_h,(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*num*sizeof(cuFloatComplex),
            cudaMemcpyHostToDevice));
    
    cuFloatComplex *mat_d;
    CUDA_CALL(cudaMalloc(&mat_d,p*p*p*p*num*sizeof(cuFloatComplex)));
    transMatsGen<<<gridStruct,blockStruct>>>(enlMat_d,num,p,mat_d);
    
    cuFloatComplex *coeff_d;
    CUDA_CALL(cudaMalloc(&coeff_d,p*p*num*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(coeff_d,coeff,p*p*num*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    cuFloatComplex *prod_d;
    CUDA_CALL(cudaMalloc(&prod_d,p*p*num*sizeof(cuFloatComplex)));
    cuMatsVecsMul<<<gridStruct,blockStruct>>>(mat_d,coeff_d,num,p*p,prod_d);
    
    CUDA_CALL(cudaMemcpy(prod,prod_d,p*p*num*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    
    CUDA_CALL(cudaFree(prod_d));
    CUDA_CALL(cudaFree(coeff_d));
    CUDA_CALL(cudaFree(mat_d));
    CUDA_CALL(cudaFree(enlMat_d));
    free(enlMat_h);
    return EXIT_SUCCESS;
}

__host__ int transMatsVecsMul_SR_rcr(const float wavNum, const cartCoord *trans, const cuFloatComplex *coeff, 
        const int num, const int p, cuFloatComplex *prod)
{
    //Rotation angles for the first rotation operator
    rotAng *rotAngle_h = (rotAng*)malloc(num*sizeof(rotAng));
    float *t_h = (float*)malloc(num*sizeof(float));
    sphCoord sphTemp;
    float temp;
    
    for(int i=0;i<num;i++) {
        sphTemp = cart2sph(trans[i]);
        rotAngle_h[i].alpha = sphTemp.phi;
        rotAngle_h[i].beta = sphTemp.theta;
        rotAngle_h[i].gamma = 0;
        t_h[i] = sphTemp.r;
    }
    
    rotAng *rotAngle_d;
    CUDA_CALL(cudaMalloc(&rotAngle_d,num*sizeof(rotAng)));
    CUDA_CALL(cudaMemcpy(rotAngle_d,rotAngle_h,num*sizeof(rotAng),cudaMemcpyHostToDevice));
    
    //Set up kernel launch parameters
    int numBlocksPerGrid, numThreadsPerBlock = 32;
    numBlocksPerGrid = (num+numThreadsPerBlock-1)/numThreadsPerBlock;
    
    dim3 gridStruct(numBlocksPerGrid,1,1);
    dim3 blockStruct(numThreadsPerBlock,1,1);
    
    //First rotation operator
    float *H_h = (float*)malloc((2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*num*sizeof(float));
    rotMatsInit(rotAngle_h,num,p,H_h);
    
    float *H_d;
    CUDA_CALL(cudaMalloc(&H_d,(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*num*sizeof(float)));
    CUDA_CALL(cudaMemcpy(H_d,H_h,(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*num*sizeof(float),cudaMemcpyHostToDevice));
    
    cuFloatComplex *rotMat_d;
    CUDA_CALL(cudaMalloc(&rotMat_d,p*p*p*p*num*sizeof(cuFloatComplex)));
    rotMatsGen<<<gridStruct,blockStruct>>>(rotAngle_d,num,p,H_d,rotMat_d);
    
    cuFloatComplex *coeff_d;
    CUDA_CALL(cudaMalloc(&coeff_d,p*p*num*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(coeff_d,coeff,p*p*num*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    cuFloatComplex *prod_d;
    CUDA_CALL(cudaMalloc(&prod_d,p*p*num*sizeof(cuFloatComplex)));
    cuRotsVecsMul<<<gridStruct,blockStruct>>>(rotMat_d,coeff_d,num,p,prod_d);
    CUDA_CALL(cudaMemcpy(coeff_d,prod_d,p*p*num*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
    
    //Coaxial translation operator
    float *t_d;
    CUDA_CALL(cudaMalloc(&t_d,num*sizeof(float)));
    CUDA_CALL(cudaMemcpy(t_d,t_h,num*sizeof(float),cudaMemcpyHostToDevice));
    
    cuFloatComplex *enlCoaxTransMat_h = (cuFloatComplex*)malloc((2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*
            num*sizeof(cuFloatComplex));
    srCoaxTransMatsInit(wavNum,t_h,num,p,enlCoaxTransMat_h);
    cuFloatComplex *enlCoaxTransMat_d;
    CUDA_CALL(cudaMalloc(&enlCoaxTransMat_d,(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*
            num*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(enlCoaxTransMat_d,enlCoaxTransMat_h,(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*
            num*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    cuFloatComplex *coaxTransMat_d;
    CUDA_CALL(cudaMalloc(&coaxTransMat_d,p*p*p*p*num*sizeof(cuFloatComplex)));
    coaxTransMatsGen<<<gridStruct,blockStruct>>>(enlCoaxTransMat_d,num,p,coaxTransMat_d);
    
    cuCoaxTransMatsVecsMul<<<gridStruct,blockStruct>>>(coaxTransMat_d,coeff_d,num,p,prod_d);
    CUDA_CALL(cudaMemcpy(coeff_d,prod_d,p*p*num*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
    
    //Second rotation operator
    for(int i=0;i<num;i++) {
        temp = rotAngle_h[i].alpha;
        rotAngle_h[i].alpha = rotAngle_h[i].gamma;
        rotAngle_h[i].gamma = temp;
    }
    rotMatsInit(rotAngle_h,num,p,H_h);
    CUDA_CALL(cudaMemcpy(H_d,H_h,(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)*num*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(rotAngle_d,rotAngle_h,num*sizeof(rotAng),cudaMemcpyHostToDevice));
    
    rotMatsGen<<<gridStruct,blockStruct>>>(rotAngle_d,num,p,H_d,rotMat_d);
    cuRotsVecsMul<<<gridStruct,blockStruct>>>(rotMat_d,coeff_d,num,p,prod_d);
    CUDA_CALL(cudaMemcpy(prod,prod_d,p*p*num*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    CUDA_CALL(cudaFree(prod_d));
    CUDA_CALL(cudaFree(coeff_d));
    CUDA_CALL(cudaFree(enlCoaxTransMat_d));
    CUDA_CALL(cudaFree(coaxTransMat_d));
    CUDA_CALL(cudaFree(H_d));
    CUDA_CALL(cudaFree(rotMat_d));
    CUDA_CALL(cudaFree(rotAngle_d));
    
    free(H_h);
    free(enlCoaxTransMat_h);
    free(t_h);
    free(rotAngle_h);
    return EXIT_SUCCESS;
}

__host__ __device__ void cuMatVecMul_rcr(const cuFloatComplex *rotMat1, const cuFloatComplex *coaxMat, 
        const cuFloatComplex *rotMat2, const cuFloatComplex *vec, const int p, cuFloatComplex *prod)
{
    cuFloatComplex *prod_1, *prod_2;
    prod_1 = (cuFloatComplex*)malloc(p*p*sizeof(cuFloatComplex));
    prod_2 = (cuFloatComplex*)malloc(p*p*sizeof(cuFloatComplex));
    
    cuRotVecMul(rotMat1,vec,p,prod_1);
    cuCoaxTransMatVecMul(coaxMat,prod_1,p,prod_2);
    cuRotVecMul(rotMat2,prod_2,p,prod);
    
    free(prod_1);
    free(prod_2);
}

__global__ void cuMatsVecsMul_rcr(const cuFloatComplex *rotMat1, const cuFloatComplex *coaxMat, 
        const cuFloatComplex *rotMat2, const cuFloatComplex *vec, const int num, const int p, 
        cuFloatComplex *prod)
{
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < num) {
        cuMatVecMul_rcr(&rotMat1[p*p*p*p*idx],&coaxMat[p*p*p*p*idx],&rotMat2[p*p*p*p*idx],
                &vec[p*p*idx],p,&prod[p*p*idx]);
    }
}

/*
//generate the sparse rotation and coaxial translation matrices
int genOctree(const char *filename, const float wavNum, const int s, octree *oct)
{   
    //pointer to points and elements
    cartCoord_d *pt;
    triElem *elem;
    
    //translation vectors
    cartCoord *transVec;
    float *coaxTransVec;
    rotAng *ang1, *ang2;
    int prntIdx;
    cartCoord_d cartTemp[2];
    sphCoord sphTemp;
    
    //matrices on host memory
    cuFloatComplex *rotMat1, *coaxMat, *rotMat2;
    
    //number of points and elements
    int numPt, numElem, p;
    double eps = 0.05;
    //find the number of points and elements and allocate memory
    findNum(filename,&numPt,&numElem);
    
    //allocate memory for points and elements
    pt = (cartCoord_d*)malloc(numPt*sizeof(cartCoord_d));
    elem = (triElem*)malloc(numElem*sizeof(triElem));
    
    //read the obj file
    HOST_CALL(readOBJ(filename,pt,elem));
    
    //boxes at the bottom level
    int *srcBoxSet = (int*)malloc((numElem+1)*sizeof(int));
    srcBoxes(pt,elem,numElem,s,srcBoxSet,&oct->lmax,&oct->d,&oct->pt_min);
    oct->lmin = 2;
    
    //Generate the octree
    oct->fmmLevelSet = (int**)malloc((oct->lmax-oct->lmin+1)*sizeof(int*));
    FMMLevelSet(srcBoxSet,oct->lmax,oct->fmmLevelSet);
    printf("FMM level sets: \n");
    printFMMLevelSet(oct->fmmLevelSet,oct->lmax);
    
    //Allocate memory for rotation and coaxial translation matrices of the ss translation
    oct->rotMat1_ss = (cuFloatComplex**)malloc((oct->lmax-oct->lmin)*sizeof(cuFloatComplex*));
    oct->coaxMat_ss = (cuFloatComplex**)malloc((oct->lmax-oct->lmin)*sizeof(cuFloatComplex*));
    oct->rotMat2_ss = (cuFloatComplex**)malloc((oct->lmax-oct->lmin)*sizeof(cuFloatComplex*));
    
    printf("Upward pass: \n");
    //generate matrices of the upward pass
    for(int l=oct->lmax;l>=oct->lmin+1;l--) {
        //compute the truncation number p at level l
        p = truncNum(wavNum,eps,1.5,pow(2,-l)*oct->d);
        printf("l = %d, p = %d\n",l,p);
        printf("Number of boxes: %d\n",(oct->fmmLevelSet[l-oct->lmin])[0]);
        float memNeed = (float)(p*(4*p*p-1)/3*2+p*(2*p*p+3*p+1)/6)*(oct->fmmLevelSet[l-oct->lmin])[0]
            *sizeof(cuFloatComplex)/(1024*1024*1024.0);
        printf("Memory need for SS translations at the current level is: %fGB\n",memNeed);
        
        //allocate host memory for sparse matrices of each level
        rotMat1 = (cuFloatComplex*)malloc((p*(4*p*p-1)/3)*(oct->fmmLevelSet[l-oct->lmin])[0]
                *sizeof(cuFloatComplex));
        coaxMat = (cuFloatComplex*)malloc((p*(2*p*p+3*p+1)/6)*(oct->fmmLevelSet[l-oct->lmin])[0]
                *sizeof(cuFloatComplex));
        rotMat2 = (cuFloatComplex*)malloc((p*(4*p*p-1)/3)*(oct->fmmLevelSet[l-oct->lmin])[0]
                *sizeof(cuFloatComplex));
        
        //allocate device memory for sparse matrices of each level
        CUDA_CALL(cudaMalloc((void**)&oct->rotMat1_ss[l-(oct->lmin+1)],
                (p*(4*p*p-1)/3)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex)));
        CUDA_CALL(cudaMalloc((void**)&oct->coaxMat_ss[l-(oct->lmin+1)],
                (p*(2*p*p+3*p+1)/6)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex)));
        CUDA_CALL(cudaMalloc((void**)&oct->rotMat2_ss[l-(oct->lmin+1)],
                (p*(4*p*p-1)/3)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex)));
        
        //allocate memory for translation vectors at level l
        transVec = (cartCoord*)malloc((oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cartCoord));
        
        //allocate memory for angles and coaxial translations
        ang1 = (rotAng*)malloc((oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(rotAng));
        coaxTransVec = (float*)malloc((oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(float));
        ang2 = (rotAng*)malloc((oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(rotAng));
        
        for(int i=0;i<(oct->fmmLevelSet[l-oct->lmin])[0];i++) {
            //get the box center of both the child and the parent boxes
            cartTemp[0] = boxCenter((oct->fmmLevelSet[l-oct->lmin])[i+1],l);
            cartTemp[0] = descale(cartTemp[0],oct->pt_min,oct->d);
            
            prntIdx = parent((oct->fmmLevelSet[l-oct->lmin])[i+1]);
            cartTemp[1] = boxCenter(prntIdx,l-1);
            cartTemp[1] = descale(cartTemp[1],oct->pt_min,oct->d);
            
            //compute the translation vector beteen the two box centers
            transVec[i] = cartCoord_d2cartCoord(cartCoordSub_d(cartTemp[1],cartTemp[0]));
            
            //compute the rcr parameters
            sphTemp = cart2sph(transVec[i]);
            ang1[i].alpha = sphTemp.phi;
            ang1[i].beta = sphTemp.theta;
            ang1[i].gamma = 0;
            
            coaxTransVec[i] = sphTemp.r;
            
            //angle 2 is in the reverse direction of angle 1 
            ang2[i].alpha = ang1[i].gamma;
            ang2[i].gamma = ang1[i].alpha;
            ang2[i].beta = ang1[i].beta;
        }
        
        //generate rotation and coaxial translation matrices on the host memory
        HOST_CALL(genSparseRotMats(ang1,(oct->fmmLevelSet[l-oct->lmin])[0],p,rotMat1));
        HOST_CALL(genSSSparseCoaxTransMat(wavNum,coaxTransVec,(oct->fmmLevelSet[l-oct->lmin])[0],p,coaxMat));
        HOST_CALL(genSparseRotMats(ang2,(oct->fmmLevelSet[l-oct->lmin])[0],p,rotMat2));
        
        //copy matrices from host memory to device memory
        CUDA_CALL(cudaMemcpy(oct->rotMat1_ss[l-(oct->lmin+1)],rotMat1,
                (p*(4*p*p-1)/3)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(oct->coaxMat_ss[l-(oct->lmin+1)],coaxMat,
                (p*(2*p*p+3*p+1)/6)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(oct->rotMat2_ss[l-(oct->lmin+1)],rotMat2,
                (p*(4*p*p-1)/3)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
        
        //release memory
        free(rotMat1);
        free(coaxMat);
        free(rotMat2);
        free(transVec);
        free(ang1);
        free(coaxTransVec);
        free(ang2);
    }
    
    //generate matrices of the download pass
    int set[27*27+1], intSet[27*27+1]; //used to save I4 sets
    
    //number of sr translations
    int **fmmSRNumSet = (int**)malloc((oct->lmax-oct->lmin+1)*sizeof(int*));
    int numSR, idx;
    FMMLevelSetNumSR(oct->fmmLevelSet,oct->lmax,fmmSRNumSet);
    printf("SR sets: \n");
    for(int l=oct->lmin;l<=oct->lmax;l++) {
        p = truncNum(wavNum,eps,1.5,pow(2,-l)*oct->d);
        numSR = 0;
        for(int i=0;i<(oct->fmmLevelSet[l-oct->lmin])[0];i++) {
            numSR+=fmmSRNumSet[l-oct->lmin][i];
            //printf("%d ",fmmSRNumSet[l-oct->lmin][i]);
        }
        printf("Total number of sr translations at level %d: %d\n",l,numSR);
        float memNeed = (float)(p*(4*p*p-1)/3*2+p*(2*p*p+3*p+1)/6)*numSR*sizeof(cuFloatComplex)/(1024*1024*1024.0);
        printf("Memory need for SR translations at the current level is: %fGB\n",memNeed);
        //printf("\n");
    }
    printf("Number of SR translations: \n");
    printLevelSetNumSR(fmmSRNumSet,oct->fmmLevelSet,oct->lmax);
    
    //allocate memory for rotation and coaxial translation matrices of the ss translation
    oct->rotMat1_sr = (cuFloatComplex**)malloc((oct->lmax-oct->lmin+1)*sizeof(cuFloatComplex*));
    oct->coaxMat_sr = (cuFloatComplex**)malloc((oct->lmax-oct->lmin+1)*sizeof(cuFloatComplex*));
    oct->rotMat2_sr = (cuFloatComplex**)malloc((oct->lmax-oct->lmin+1)*sizeof(cuFloatComplex*));
    
    printf("Downward pass, SR: \n");
    //go down from level lmin to level lmax
    for(int l=oct->lmin;l<=oct->lmax;l++) {
        //compute the truncation number p at level l
        p = truncNum(wavNum,eps,1.5,pow(2,-l)*oct->d);
        printf("l = %d, p = %d\n",l,p);
        
        numSR = 0;
        idx = 0; //index for the translation vector
        for(int i=0;i<(oct->fmmLevelSet[l-oct->lmin])[0];i++) {
            numSR += (fmmSRNumSet[l-oct->lmin])[i];
        }
        printf("numSR: %d\n",numSR);
        
        //allocate host memory for rotation matrices and coaxial translation matrices
        rotMat1 = (cuFloatComplex*)malloc((p*(4*p*p-1)/3)*numSR*sizeof(cuFloatComplex));
        coaxMat = (cuFloatComplex*)malloc((p*(2*p*p+3*p+1)/6)*numSR*sizeof(cuFloatComplex));
        rotMat2 = (cuFloatComplex*)malloc((p*(4*p*p-1)/3)*numSR*sizeof(cuFloatComplex));
        
        //allocate device memory for sparse rotation matrices and coaxial translation matrices
        CUDA_CALL(cudaMalloc((void**)&oct->rotMat1_sr[l-oct->lmin],(p*(4*p*p-1)/3)*numSR*sizeof(cuFloatComplex)));
        CUDA_CALL(cudaMalloc((void**)&oct->coaxMat_sr[l-oct->lmin],(p*(2*p*p+3*p+1)/6)*numSR*sizeof(cuFloatComplex)));
        CUDA_CALL(cudaMalloc((void**)&oct->rotMat2_sr[l-oct->lmin],(p*(4*p*p-1)/3)*numSR*sizeof(cuFloatComplex)));
        
        //allocate memory for translation vectors and angles and coaxial translation vectors
        transVec = (cartCoord*)malloc(numSR*sizeof(cartCoord));
        
        ang1 = (rotAng*)malloc(numSR*sizeof(rotAng));
        coaxTransVec = (float*)malloc(numSR*sizeof(float));
        ang2 = (rotAng*)malloc(numSR*sizeof(rotAng));
        
        //compute the translation vectors at level l
        for(int i=0;i<(oct->fmmLevelSet[l-oct->lmin])[0];i++) {
            //get the I4 set of the current box
            I4((oct->fmmLevelSet[l-oct->lmin])[i+1],l,set);
            intersection(set,oct->fmmLevelSet[l-oct->lmin],intSet);
            
            //get the center of the current box
            cartTemp[1] = boxCenter((oct->fmmLevelSet[l-oct->lmin])[i+1],l);
            cartTemp[1] = descale(cartTemp[1],oct->pt_min,oct->d);
            for(int j=0;j<intSet[0];j++) {
                //get the center of the starting box
                cartTemp[0] = boxCenter(intSet[j+1],l);
                cartTemp[0] = descale(cartTemp[0],oct->pt_min,oct->d);
                
                //translation vector
                transVec[idx] = cartCoord_d2cartCoord(cartCoordSub_d(cartTemp[1],cartTemp[0]));
                
                //compute the rcr parameters
                sphTemp = cart2sph(transVec[idx]);
                ang1[idx].alpha = sphTemp.phi;
                ang1[idx].beta = sphTemp.theta;
                ang1[idx].gamma = 0;

                coaxTransVec[idx] = sphTemp.r;

                ang2[idx].alpha = ang1[i].gamma;
                ang2[idx].gamma = ang1[i].alpha;
                ang2[idx].beta = ang1[i].beta;
                
                idx++;
            }
        }
        
        HOST_CALL(genSparseRotMats(ang1,numSR,p,rotMat1));
        HOST_CALL(genSRSparseCoaxTransMat(wavNum,coaxTransVec,numSR,p,coaxMat));
        HOST_CALL(genSparseRotMats(ang2,numSR,p,rotMat2));
        
        CUDA_CALL(cudaMemcpy(oct->rotMat1_sr[l-oct->lmin],rotMat1,
                (p*(4*p*p-1)/3)*numSR*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(oct->coaxMat_sr[l-oct->lmin],coaxMat,
                (p*(2*p*p+3*p+1)/6)*numSR*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(oct->rotMat2_sr[l-oct->lmin],rotMat2,
                (p*(4*p*p-1)/3)*numSR*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
        
        //release memory
        free(rotMat1);
        free(coaxMat);
        free(rotMat2);
        free(transVec);
        free(ang1);
        free(coaxTransVec);
        free(ang2);
        free(fmmSRNumSet[l-oct->lmin]);
    }
    
    free(fmmSRNumSet);
    
    //set up rr translation matrices
    oct->rotMat1_rr = (cuFloatComplex**)malloc((oct->lmax-oct->lmin)*sizeof(cuFloatComplex*));
    oct->coaxMat_rr = (cuFloatComplex**)malloc((oct->lmax-oct->lmin)*sizeof(cuFloatComplex*));
    oct->rotMat2_rr = (cuFloatComplex**)malloc((oct->lmax-oct->lmin)*sizeof(cuFloatComplex*));
    
    printf("Downward pass, RR: \n");
    for(int l=oct->lmin+1;l<=oct->lmax;l++) {
        //determine the truncation number
        p = truncNum(wavNum,eps,1.5,pow(2,-l)*oct->d);
        printf("l = %d, p = %d\n",l,p);
        float memNeed = (float)(p*(4*p*p-1)/3*2+p*(2*p*p+3*p+1)/6)*(oct->fmmLevelSet[l-oct->lmin])[0]
            *sizeof(cuFloatComplex)/(1024*1024*1024.0);
        printf("Memory need for RR translations at the current level is: %fGB\n",memNeed);
        
        //allocate host memory for rotation matrices and coaxial translation matrices
        rotMat1 = (cuFloatComplex*)malloc((p*(4*p*p-1)/3)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex));
        coaxMat = (cuFloatComplex*)malloc((p*(2*p*p+3*p+1)/6)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex));
        rotMat2 = (cuFloatComplex*)malloc((p*(4*p*p-1)/3)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex));
        
        //allocate memory for rotation matrices and coaxial translation matrices
        CUDA_CALL(cudaMalloc((void**)&oct->rotMat1_rr[l-(oct->lmin+1)],
                (p*(4*p*p-1)/3)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex)));
        CUDA_CALL(cudaMalloc((void**)&oct->coaxMat_rr[l-(oct->lmin+1)],
                (p*(2*p*p+3*p+1)/6)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex)));
        CUDA_CALL(cudaMalloc((void**)&oct->rotMat2_rr[l-(oct->lmin+1)],
                (p*(4*p*p-1)/3)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex)));
        
        //allocate memory for translation vectors and angles and coaxial translation vectors
        transVec = (cartCoord*)malloc((oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cartCoord));
        
        ang1 = (rotAng*)malloc((oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(rotAng));
        coaxTransVec = (float*)malloc((oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(float));
        ang2 = (rotAng*)malloc((oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(rotAng));
        
        //compute the translation vectors
        for(int i=0;i<(oct->fmmLevelSet[l-oct->lmin])[0];i++) {
            //get the box center of both the child and the parent boxes
            cartTemp[1] = boxCenter((oct->fmmLevelSet[l-oct->lmin])[i+1],l);
            cartTemp[1] = descale(cartTemp[1],oct->pt_min,oct->d);
            
            prntIdx = parent((oct->fmmLevelSet[l-oct->lmin])[i+1]);
            cartTemp[0] = boxCenter(prntIdx,l-1);
            cartTemp[0] = descale(cartTemp[0],oct->pt_min,oct->d);
            
            //compute the translation vector beteen the two box centers
            transVec[i] = cartCoord_d2cartCoord(cartCoordSub_d(cartTemp[1],cartTemp[0]));
            
            //compute the rcr parameters
            sphTemp = cart2sph(transVec[i]);
            
            ang1[i].alpha = sphTemp.phi;
            ang1[i].beta = sphTemp.theta;
            ang1[i].gamma = 0;
            
            coaxTransVec[i] = sphTemp.r;
            
            ang2[i].alpha = ang1[i].gamma;
            ang2[i].gamma = ang1[i].alpha;
            ang2[i].beta = ang1[i].beta;
        }
        
        //generate rotation and coaxial translation matrices on the host memory
        HOST_CALL(genSparseRotMats(ang1,(oct->fmmLevelSet[l-oct->lmin])[0],p,rotMat1));
        HOST_CALL(genRRSparseCoaxTransMat(wavNum,coaxTransVec,(oct->fmmLevelSet[l-oct->lmin])[0],p,coaxMat));
        HOST_CALL(genSparseRotMats(ang2,(oct->fmmLevelSet[l-oct->lmin])[0],p,rotMat2));
        
        //copy matrices from host memory to device memory
        CUDA_CALL(cudaMemcpy(oct->rotMat1_rr[l-(oct->lmin+1)],rotMat1,
                (p*(4*p*p-1)/3)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(oct->coaxMat_rr[l-(oct->lmin+1)],coaxMat,
                (p*(2*p*p+3*p+1)/6)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(oct->rotMat2_rr[l-(oct->lmin+1)],rotMat2,
                (p*(4*p*p-1)/3)*(oct->fmmLevelSet[l-oct->lmin])[0]*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
        
        //release memory
        free(rotMat1);
        free(coaxMat);
        free(rotMat2);
        free(transVec);
        free(ang1);
        free(coaxTransVec);
        free(ang2);
    }
    
    return EXIT_SUCCESS;
}

int destroyOctree(octree *oct, const int lmax)
{
    for(int l=oct->lmin;l<=lmax;l++) {
        CUDA_CALL(cudaFree(oct->rotMat1_sr[l-oct->lmin]));
        CUDA_CALL(cudaFree(oct->coaxMat_sr[l-oct->lmin]));
        CUDA_CALL(cudaFree(oct->rotMat2_sr[l-oct->lmin]));
        if(l!=oct->lmin) {
            CUDA_CALL(cudaFree(oct->rotMat1_ss[l-(oct->lmin+1)]));
            CUDA_CALL(cudaFree(oct->coaxMat_ss[l-(oct->lmin+1)]));
            CUDA_CALL(cudaFree(oct->rotMat2_ss[l-(oct->lmin+1)]));
            
            CUDA_CALL(cudaFree(oct->rotMat1_rr[l-(oct->lmin+1)]));
            CUDA_CALL(cudaFree(oct->coaxMat_rr[l-(oct->lmin+1)]));
            CUDA_CALL(cudaFree(oct->rotMat2_rr[l-(oct->lmin+1)]));
        }
        free(oct->fmmLevelSet[l-2]);
    }
    free(oct->fmmLevelSet);
    return EXIT_SUCCESS;
}
 */

__host__ int testSparseRotMatsGen(const rotAng *rotAngle, const int numRot, const int p)
{
    //allocate memory for the products
    cuFloatComplex *prod_h = (cuFloatComplex*)malloc(numRot*p*p*sizeof(cuFloatComplex));
    cuFloatComplex *prod_d;
    CUDA_CALL(cudaMalloc(&prod_d,numRot*p*p*sizeof(cuFloatComplex)));
    
    //generate random vectors for multiplication
    cuFloatComplex *vec_h = (cuFloatComplex*)malloc(numRot*p*p*sizeof(cuFloatComplex));
    HOST_CALL(genRndCoeffs(numRot*p*p,vec_h));
    
    //copy the vectors to the device memory
    cuFloatComplex *vec_d;
    CUDA_CALL(cudaMalloc(&vec_d,numRot*p*p*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(vec_d,vec_h,numRot*p*p*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    //generate sparse rotation matrices
    cuFloatComplex *sparseRotMat_h = (cuFloatComplex*)malloc(numRot*(p*(4*p*p-1)/3)*sizeof(cuFloatComplex));
    HOST_CALL(genSparseRotMats(rotAngle,numRot,p,sparseRotMat_h));
    
    //allocate device memory to save the sparse rotation matrices
    cuFloatComplex *sparseRotMat_d;
    CUDA_CALL(cudaMalloc(&sparseRotMat_d,numRot*(p*(4*p*p-1)/3)*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(sparseRotMat_d,sparseRotMat_h,numRot*(p*(4*p*p-1)/3)*sizeof(cuFloatComplex),
            cudaMemcpyHostToDevice));
    
    int numBlocksPerGrid, numThreadsPerBlock = 32;
    numBlocksPerGrid = (numRot+numThreadsPerBlock-1)/numThreadsPerBlock;
    dim3 gridStruct(numBlocksPerGrid,1,1);
    dim3 blockStruct(numThreadsPerBlock,1,1);
    
    cuSparseRotsVecsMul<<<gridStruct,blockStruct>>>(sparseRotMat_d,vec_d,numRot,p,prod_d);
    CUDA_CALL(cudaMemcpy(prod_h,prod_d,numRot*p*p*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    //print the products
    printf("Sparse: \n");
    for(int i=0;i<numRot;i++) {
        printMat_cuFloatComplex(&prod_h[i*p*p],1,p*p,1);
    }
    
    cuFloatComplex *denseRotMat_h = (cuFloatComplex*)malloc(numRot*p*p*p*p*sizeof(cuFloatComplex));
    HOST_CALL(genRotMats(rotAngle,numRot,p,denseRotMat_h));
    cuFloatComplex *denseRotMat_d;
    CUDA_CALL(cudaMalloc(&denseRotMat_d,numRot*p*p*p*p*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(denseRotMat_d,denseRotMat_h,numRot*p*p*p*p*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    cuMatsVecsMul<<<gridStruct,blockStruct>>>(denseRotMat_d,vec_d,numRot,p*p,prod_d);
    CUDA_CALL(cudaMemcpy(prod_h,prod_d,numRot*p*p*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    printf("Dense: \n");
    for(int i=0;i<numRot;i++) {
        printMat_cuFloatComplex(&prod_h[i*p*p],1,p*p,1);
    }
    
    free(denseRotMat_h);
    free(sparseRotMat_h);
    free(vec_h);
    free(prod_h);
    
    CUDA_CALL(cudaFree(sparseRotMat_d));
    CUDA_CALL(cudaFree(denseRotMat_d));
    CUDA_CALL(cudaFree(vec_d));
    CUDA_CALL(cudaFree(prod_d));
    
    return EXIT_SUCCESS;
}

__host__ int testSparseCoaxTransMatsGen(const float wavNum, const float *transVec, const int numTransVec, 
        const int p)
{
    //allocate memory for the products
    cuFloatComplex *prod_h = (cuFloatComplex*)malloc(numTransVec*p*p*sizeof(cuFloatComplex));
    cuFloatComplex *prod_d;
    CUDA_CALL(cudaMalloc(&prod_d,numTransVec*p*p*sizeof(cuFloatComplex)));
    
    //generate random vectors for multiplication
    cuFloatComplex *vec_h = (cuFloatComplex*)malloc(numTransVec*p*p*sizeof(cuFloatComplex));
    HOST_CALL(genRndCoeffs(numTransVec*p*p,vec_h));
    
    //copy the vectors to the device memory
    cuFloatComplex *vec_d;
    CUDA_CALL(cudaMalloc(&vec_d,numTransVec*p*p*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(vec_d,vec_h,numTransVec*p*p*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    //generate sparse coaxial translation matrices
    cuFloatComplex *sparseMat_h = (cuFloatComplex*)malloc(numTransVec*(p*(2*p*p+3*p+1)/6)*sizeof(cuFloatComplex));
    HOST_CALL(genRRSparseCoaxTransMat(wavNum,transVec,numTransVec,p,sparseMat_h));
    
    //allocate device memory to save the sparse rotation matrices
    cuFloatComplex *sparseMat_d;
    CUDA_CALL(cudaMalloc(&sparseMat_d,numTransVec*(p*(2*p*p+3*p+1)/6)*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(sparseMat_d,sparseMat_h,numTransVec*(p*(2*p*p+3*p+1)/6)*sizeof(cuFloatComplex),
            cudaMemcpyHostToDevice));
    
    int numBlocksPerGrid, numThreadsPerBlock = 32;
    numBlocksPerGrid = (numTransVec+numThreadsPerBlock-1)/numThreadsPerBlock;
    dim3 gridStruct(numBlocksPerGrid,1,1);
    dim3 blockStruct(numThreadsPerBlock,1,1);
    
    cuSparseCoaxTransMatsVecsMul<<<gridStruct,blockStruct>>>(sparseMat_d,vec_d,numTransVec,p,prod_d);
    CUDA_CALL(cudaMemcpy(prod_h,prod_d,numTransVec*p*p*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    //print the products
    printf("Sparse: \n");
    for(int i=0;i<numTransVec;i++) {
        printMat_cuFloatComplex(&prod_h[i*p*p],1,p*p,1);
    }
    
    cuFloatComplex *denseMat_h = (cuFloatComplex*)malloc(numTransVec*p*p*p*p*sizeof(cuFloatComplex));
    HOST_CALL(genRRCoaxTransMat(wavNum,transVec,numTransVec,p,denseMat_h));
    cuFloatComplex *denseMat_d;
    CUDA_CALL(cudaMalloc(&denseMat_d,numTransVec*p*p*p*p*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(denseMat_d,denseMat_h,numTransVec*p*p*p*p*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    cuMatsVecsMul<<<gridStruct,blockStruct>>>(denseMat_d,vec_d,numTransVec,p*p,prod_d);
    CUDA_CALL(cudaMemcpy(prod_h,prod_d,numTransVec*p*p*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    printf("Dense: \n");
    for(int i=0;i<numTransVec;i++) {
        printMat_cuFloatComplex(&prod_h[i*p*p],1,p*p,1);
    }
    
    free(denseMat_h);
    free(sparseMat_h);
    free(vec_h);
    free(prod_h);
    
    CUDA_CALL(cudaFree(sparseMat_d));
    CUDA_CALL(cudaFree(denseMat_d));
    CUDA_CALL(cudaFree(vec_d));
    CUDA_CALL(cudaFree(prod_d));
    
    return EXIT_SUCCESS;
}

__host__ void genSRCoaxTransVecsRotAngles(const int l, const double d, const cartCoord_d pt_min, 
        float **pVec, int *pNumVec, rotAng **pRotAngle, int *pNumRotAng)
{
    //set both numbers of vectors and angles to zero
    int idx, numVec = 0, numRot = 0;
    int set[8*27+1];
    cartCoord_d boxCtr[2], vec;
    sphCoord coord_sph;
    rotAng *tempAng, angle;
    float *tempVec, t, eps = 0.00001*d;
    bool flag;
    
    if(l==2) {
        //allocate memory for angles and vectors
        tempAng = (rotAng*)malloc((64-27)*64*sizeof(rotAng));
        tempVec = (float*)malloc((64-27)*64*sizeof(float));
        //The level is 2
        for(idx=0;idx<(int)pow(8,l);idx++) {
            //the coordinate of the terminating point
            boxCtr[1] = boxCenter(idx,l);
            boxCtr[1] = descale(boxCtr[1],pt_min,d);
            //get the I4 set of the current index
            I4(idx,l,set);
            for(int i=0;i<set[0];i++) {
                //the coordinate of the starting point
                boxCtr[0] = boxCenter(set[i+1],l);
                boxCtr[0] = descale(boxCtr[0],pt_min,d);
                vec = cartCoordSub_d(boxCtr[1],boxCtr[0]);
                coord_sph = cart2sph(cartCoord_d2cartCoord(vec));
                angle.alpha = coord_sph.phi;
                angle.beta = coord_sph.theta;
                angle.gamma = 0;
                t = coord_sph.r;
                //printf("t: %f\n",t);
                
                //update tempVec
                if(numVec==0) {
                    tempVec[numVec++] = t;
                } else {
                    flag = false; //not in existing vectors
                    for(int j=0;j<numVec;j++) {
                        //tell if t is in exising vectors
                        //printf("abs(tempVec[j]-t): %f\n",abs(tempVec[j]-t));
                        if(abs(tempVec[j]-t)<eps) {
                            flag = true;
                            break;
                        }
                    }
                    if(!flag) {
                        tempVec[numVec++] = t;
                    }
                }
                
                //update tempAng
                if(numRot==0) {
                    tempAng[numRot++] = angle;
                } else {
                    flag = false;
                    for(int j=0;j<numRot;j++) {
                        //tell if rotAng is in existing rotation angles
                        if(abs(angle.alpha-tempAng[j].alpha)<eps && abs(angle.beta-tempAng[j].beta)<eps) {
                            flag = true;
                            break;
                        }
                    }
                    if(!flag) {
                        tempAng[numRot++] = angle;
                    }
                }
            }
            
        }
    } else {
        //allocate memory for angles and vectors
        tempVec = (float*)malloc(208*8*sizeof(float));
        tempAng = (rotAng*)malloc(208*8*sizeof(rotAng));
        
        //find the first box with the largest number of sr translations
        for(idx=0;idx<(int)pow(8,l);idx++) {
            I4(idx,l,set);
            if(set[0]==189) {
                break;
            }
        }
        //printf("idx: %d\n",idx);
        
        //get the parent box of the current box
        int prntIdx = parent(idx);
        
        //iterate through all child boxes of the parent box
        for(int childIdx=0;childIdx<8;childIdx++) {
            //get the index of the child box
            idx = child(prntIdx,childIdx);
            
            //get the terminating point
            boxCtr[1] = boxCenter(idx,l);
            boxCtr[1] = descale(boxCtr[1],pt_min,d);
            
            //get the I4 set of the current box
            I4(idx,l,set);
            for(int i=0;i<set[0];i++) {
                //get the starting point
                boxCtr[0] = boxCenter(set[i+1],l);
                boxCtr[0] = descale(boxCtr[0],pt_min,d);
                
                //vector from the starting point to the terminating point
                vec = cartCoordSub_d(boxCtr[1],boxCtr[0]);
                coord_sph = cart2sph(cartCoord_d2cartCoord(vec));
                angle.alpha = coord_sph.phi;
                angle.beta = coord_sph.theta;
                angle.gamma = 0;
                t = coord_sph.r;
                
                //update tempVec
                if(numVec==0) {
                    tempVec[numVec++] = t;
                } else {
                    flag = false; //not in existing vectors
                    for(int j=0;j<numVec;j++) {
                        //tell if t is in exising vectors
                        if(abs(tempVec[j]-t)<eps) {
                            flag = true;
                            break;
                        }
                    }
                    if(!flag) {
                        tempVec[numVec++] = t;
                    }
                }
                
                //update tempAng
                if(numRot==0) {
                    tempAng[numRot++] = angle;
                } else {
                    flag = false;
                    for(int j=0;j<numRot;j++) {
                        //tell if rotAng is in existing rotation angles
                        if(abs(angle.alpha-tempAng[j].alpha)<eps && abs(angle.beta-tempAng[j].beta)<eps) {
                            flag = true;
                            break;
                        }
                    }
                    if(!flag) {
                        tempAng[numRot++] = angle;
                    }
                }
            }
        }
    }
    
    //update the number of vectors and the number of rotations at level l
    *pNumVec = numVec;
    *pNumRotAng = numRot;
    *pVec = (float*)malloc(numVec*sizeof(float));
    *pRotAngle = (rotAng*)malloc(numRot*sizeof(rotAng));
    for(int i=0;i<numVec;i++) {
        (*pVec)[i] = tempVec[i];
    }
    for(int i=0;i<numRot;i++) {
        (*pRotAngle)[i] = tempAng[i];
    }
    free(tempAng);
    free(tempVec);
}

__host__ void genSSCoaxTransVecsRotAngles(const int l, const double d, const cartCoord_d pt_min, 
        float **pVec, int *pNumVec, rotAng **pRotAngle, int *pNumRotAng)
{
    int prntIdx = parent(0), idx, numVec = 0, numRot = 0;
    bool flag;
    cartCoord_d boxCtr[2], vec;
    rotAng angle, *tempAng;
    float t, *tempVec, eps = 0.000001*d;
    sphCoord coord_sph;
    tempAng = (rotAng*)malloc(8*sizeof(rotAng));
    tempVec = (float*)malloc(8*sizeof(float));
    
    boxCtr[1] = boxCenter(prntIdx,l-1);
    boxCtr[1] = descale(boxCtr[1],pt_min,d);
    for(int childIdx=0;childIdx<8;childIdx++) {
        idx = child(prntIdx,childIdx);
        boxCtr[0] = boxCenter(idx,l);
        boxCtr[0] = descale(boxCtr[0],pt_min,d);
        
        vec = cartCoordSub_d(boxCtr[1],boxCtr[0]);
        
        coord_sph = cart2sph(cartCoord_d2cartCoord(vec));
        angle.alpha = coord_sph.phi;
        angle.beta = coord_sph.theta;
        angle.gamma = 0;
        t = coord_sph.r;
        
        if(numVec==0) {
            tempVec[numVec++] = t;
        } else {
            flag = false; //not in existing vectors
            for(int j=0;j<numVec;j++) {
                //tell if t is in exising vectors
                if(abs(tempVec[j]-t)<eps) {
                    flag = true;
                    break;
                }
            }
            if(!flag) {
                tempVec[numVec++] = t;
            }
        }

        //update tempAng
        if(numRot==0) {
            tempAng[numRot++] = angle;
        } else {
            flag = false;
            for(int j=0;j<numRot;j++) {
                //tell if rotAng is in existing rotation angles
                if(abs(angle.alpha-tempAng[j].alpha)<eps && abs(angle.beta-tempAng[j].beta)<eps) {
                    flag = true;
                    break;
                }
            }
            if(!flag) {
                tempAng[numRot++] = angle;
            }
        }
    }
    
    *pNumVec = numVec;
    *pNumRotAng = numRot;
    
    *pVec = (float*)malloc(numVec*sizeof(float));
    for(int i=0;i<numVec;i++) {
        (*pVec)[i] = tempVec[i];
    }
    *pRotAngle = (rotAng*)malloc(numRot*sizeof(rotAng));
    for(int i=0;i<numRot;i++) {
        (*pRotAngle)[i] = tempAng[i];
    }
    free(tempAng);
    free(tempVec);
}

__host__ void genRRCoaxTransVecsRotAngles(const int l, const double d, const cartCoord_d pt_min, 
        float **pVec, int *pNumVec, rotAng **pRotAngle, int *pNumRotAng)
{
    //l should be larger than lmin
    
    int prntIdx = parent(0), idx, numVec = 0, numRot = 0;
    bool flag;
    cartCoord_d boxCtr[2], vec;
    rotAng angle, *tempAng;
    float t, *tempVec, eps = 0.00001*d;
    sphCoord coord_sph;
    tempAng = (rotAng*)malloc(8*sizeof(rotAng));
    tempVec = (float*)malloc(8*sizeof(float));
    boxCtr[0] = boxCenter(prntIdx,l-1);
    boxCtr[0] = descale(boxCtr[0],pt_min,d);
    
    for(int childIdx=0;childIdx<8;childIdx++) {
        idx = child(prntIdx,childIdx);
        boxCtr[1] = boxCenter(idx,l);
        boxCtr[1] = descale(boxCtr[1],pt_min,d);
        
        vec = cartCoordSub_d(boxCtr[1],boxCtr[0]);
        coord_sph = cart2sph(cartCoord_d2cartCoord(vec));
        angle.alpha = coord_sph.phi;
        angle.beta = coord_sph.theta;
        angle.gamma = 0;
        t = coord_sph.r;
        
        if(numVec==0) {
            tempVec[numVec++] = t;
        } else {
            flag = false; //not in existing vectors
            for(int j=0;j<numVec;j++) {
                //tell if t is in exising vectors
                if(abs(tempVec[j]-t)<eps) {
                    flag = true;
                    break;
                }
            }
            if(!flag) {
                tempVec[numVec++] = t;
            }
        }

        //update tempAng
        if(numRot==0) {
            tempAng[numRot++] = angle;
        } else {
            flag = false;
            for(int j=0;j<numRot;j++) {
                //tell if rotAng is in existing rotation angles
                if(abs(angle.alpha-tempAng[j].alpha)<eps && abs(angle.beta-tempAng[j].beta)<eps) {
                    flag = true;
                    break;
                }
            }
            if(!flag) {
                tempAng[numRot++] = angle;
            }
        }
    }
    
    *pNumVec = numVec;
    *pNumRotAng = numRot;
    
    *pVec = (float*)malloc(numVec*sizeof(float));
    for(int i=0;i<numVec;i++) {
        (*pVec)[i] = tempVec[i];
    }
    *pRotAngle = (rotAng*)malloc(numRot*sizeof(rotAng));
    for(int i=0;i<numRot;i++) {
        (*pRotAngle)[i] = tempAng[i];
    }
    free(tempAng);
    free(tempVec);
    
}

__host__ void initOctree(octree *oct)
{
    oct->lmin = 2;
    
    oct->numRotAng = 0;
    oct->ang = NULL;
    oct->rotMat1 = NULL;
    oct->rotMat2 = NULL;
    
    oct->numRRCoaxTransVec = 0;
    oct->rrCoaxTransVec = NULL;
    oct->rrCoaxMat = NULL;
    
    oct->numSRCoaxTransVec = 0;
    oct->srCoaxTransVec = NULL;
    oct->srCoaxMat = NULL;
    
    oct->eps = 0.01;
    oct->maxWavNum = 2*PI*20000/343.0f;
}

__host__ int genOctree(const char *filename, const float wavNum, const int s, octree *oct)
{
    //pointer to points and elements
    cartCoord_d *pt;
    triElem *elem;
    
    //number of points and elements
    int numPt, numElem;
    //find the number of points and elements and allocate memory
    findNum(filename,&numPt,&numElem);
    
    //allocate memory for points and elements
    pt = (cartCoord_d*)malloc(numPt*sizeof(cartCoord_d));
    elem = (triElem*)malloc(numElem*sizeof(triElem));
    
    //read the obj file
    HOST_CALL(readOBJ(filename,pt,elem));
    
    //boxes at the bottom level
    int *srcBoxSet = (int*)malloc((numElem+1)*sizeof(int));
    srcBoxes(pt,elem,numElem,s,srcBoxSet,&oct->lmax,&oct->d,&oct->pt_min);
    printf("lmax = %d\n",oct->lmax);
    
    oct->fmmLevelSet = (int**)malloc((oct->lmax-oct->lmin+1)*sizeof(int*));
    FMMLevelSet(srcBoxSet,oct->lmax,oct->fmmLevelSet);
    printf("successfully generated level related sets.\n");
    
    int pmax = truncNum(oct->maxWavNum,oct->eps,1.5,pow(2,-oct->lmin)*oct->d);
    float epsilon = 0.000000001*oct->d;
    
    //declare pointers for rotation angles and translation vectors
    rotAng *ang = NULL;
    float *srCoaxTransVec = NULL, *rrCoaxTransVec = NULL, *tempVec = NULL;
    int numRot, numVec;
    
    bool flag;
    
    for(int l=oct->lmin;l<=oct->lmax;l++) {
        //generate vectors and rotation angles at level l
        genSRCoaxTransVecsRotAngles(l,oct->d,oct->pt_min,&srCoaxTransVec,&numVec,&ang,&numRot);
        
        //no angles saved yet
        if(oct->numRotAng==0) {
            //allocate memory for angles and move the angles to the octree structure
            oct->ang = (rotAng*)malloc(numRot*sizeof(rotAng));
            for(int i=0;i<numRot;i++) {
                oct->ang[i] = ang[i];
            }
            
            //update the number of rotation angles
            oct->numRotAng = numRot;
        }
        
        //no vectors saved yet
        if(oct->numSRCoaxTransVec==0) {
            //allocate memory for sr coaxial translation vectors
            oct->srCoaxTransVec = (float*)malloc(numVec*sizeof(float));
            for(int i=0;i<numVec;i++) {
                oct->srCoaxTransVec[i] = srCoaxTransVec[i];
            }
            oct->numSRCoaxTransVec = numVec;
        } else {
            //allocate memory for temporary vector array
            
            tempVec = (float*)malloc((oct->numSRCoaxTransVec+numVec)*sizeof(float));
            
            //move all vectors in the array to the temporary array
            for(int i=0;i<oct->numSRCoaxTransVec;i++) {
                tempVec[i] = oct->srCoaxTransVec[i];
            }
            
            //free the memory
            free(oct->srCoaxTransVec);
            
            //tell one by one if the new vectors belong to the existing array
            for(int i=0;i<numVec;i++) {
                //assume that the new vector does not belong to the existing array
                flag = false;
                for(int j=0;j<oct->numSRCoaxTransVec;j++) {
                    if(abs(srCoaxTransVec[i]-tempVec[j])<epsilon) {
                        //the current new vector is in the old array
                        flag = true;
                        break;
                    }
                }
                if(!flag) {
                    //push the vector into the vector array and increase the number of SR vectors
                    tempVec[oct->numSRCoaxTransVec++] = srCoaxTransVec[i];
                }
            }
            
            //allocate memory for sr translations
            oct->srCoaxTransVec = (float*)malloc(oct->numSRCoaxTransVec*sizeof(float));
            for(int i=0;i<oct->numSRCoaxTransVec;i++) {
                oct->srCoaxTransVec[i] = tempVec[i];
            }
            free(tempVec);
        }
        
        free(srCoaxTransVec);
        free(ang);
    }
    
    //sort the rotation angles and the coaxial vectors
    sortRotArray(oct->ang,oct->numRotAng,oct->eps);
    bubbleSort(oct->srCoaxTransVec,oct->numSRCoaxTransVec);
    
    //generate rotation matrices, only generate one time for the largest p, unrelated to the wave number
    if(oct->rotMat1==NULL) {
        oct->rotMat1 = (cuFloatComplex*)malloc(oct->numRotAng*(pmax*(4*pmax*pmax-1)/3)*sizeof(cuFloatComplex));
        HOST_CALL(genSparseRotMats(oct->ang,oct->numRotAng,pmax,oct->rotMat1));
        
        //generate rotMat2
        ang = (rotAng*)malloc(oct->numRotAng*sizeof(rotAng));
        for(int i=0;i<oct->numRotAng;i++) {
            ang[i].alpha = oct->ang[i].gamma;
            ang[i].beta = oct->ang[i].beta;
            ang[i].gamma = oct->ang[i].alpha;
        }
        oct->rotMat2 = (cuFloatComplex*)malloc(oct->numRotAng*(pmax*(4*pmax*pmax-1)/3)*sizeof(cuFloatComplex));
        HOST_CALL(genSparseRotMats(ang,oct->numRotAng,pmax,oct->rotMat2));
        free(ang);
    }
    
    //generate coaxial translation matrices; the largest matrices corresponding to the current wave number will be generated
    if(oct->srCoaxMat!=NULL) {
        free(oct->srCoaxMat);
    }
    
    //compute the largest truncation number
    pmax = truncNum(wavNum,oct->eps,1.5,pow(2,-oct->lmin)*oct->d);
    oct->srCoaxMat = (cuFloatComplex*)malloc(oct->numSRCoaxTransVec*(pmax*(2*pmax*pmax+3*pmax+1)/6)*sizeof(cuFloatComplex));
    HOST_CALL(genSRSparseCoaxTransMat(wavNum,oct->srCoaxTransVec,oct->numSRCoaxTransVec,pmax,oct->srCoaxMat));
    
    //generate ss coaxial translations
    for(int l=oct->lmax;l>oct->lmin;l--) {
        genSSCoaxTransVecsRotAngles(l,oct->d,oct->pt_min,&rrCoaxTransVec,&numVec,&ang,&numRot);
        
        //no vectors saved yet
        if(oct->numRRCoaxTransVec==0) {
            //allocate memory for sr coaxial translation vectors
            oct->rrCoaxTransVec = (float*)malloc(numVec*sizeof(float));
            for(int i=0;i<numVec;i++) {
                oct->rrCoaxTransVec[i] = rrCoaxTransVec[i];
            }
            oct->numRRCoaxTransVec = numVec;
        } else {
            //allocate memory for temporary vector array
            tempVec = (float*)malloc((oct->numRRCoaxTransVec+numVec)*sizeof(float));
            
            //move all vectors in the array to the temporary array
            for(int i=0;i<oct->numRRCoaxTransVec;i++) {
                tempVec[i] = oct->rrCoaxTransVec[i];
            }
            
            //free the memory
            free(oct->rrCoaxTransVec);
            
            //tell one by one if the new vectors belong to the existing array
            for(int i=0;i<numVec;i++) {
                //assume that the new vector does not belong to the existing array
                flag = false;
                for(int j=0;j<oct->numRRCoaxTransVec;j++) {
                    if(abs(rrCoaxTransVec[i]-tempVec[j])<epsilon) {
                        //the current new vector is in the old array
                        flag = true;
                        break;
                    }
                }
                if(!flag) {
                    //push the vector into the vector array and increase the number of SR vectors
                    tempVec[oct->numRRCoaxTransVec++] = rrCoaxTransVec[i];
                }
            }
            
            //allocate memory for rr/ss translations
            oct->rrCoaxTransVec = (float*)malloc(oct->numRRCoaxTransVec*sizeof(float));
            for(int i=0;i<oct->numRRCoaxTransVec;i++) {
                oct->rrCoaxTransVec[i] = tempVec[i];
            }
            free(tempVec);
        }
        free(rrCoaxTransVec);
        free(ang);
    }
    
    //generate rr coaxial translations
    for(int l=oct->lmin+1;l<=oct->lmax;l++) {
        genRRCoaxTransVecsRotAngles(l,oct->d,oct->pt_min,&rrCoaxTransVec,&numVec,&ang,&numRot);
        
        //no vectors saved yet
        if(oct->numRRCoaxTransVec==0) {
            //allocate memory for sr coaxial translation vectors
            oct->rrCoaxTransVec = (float*)malloc(numVec*sizeof(float));
            for(int i=0;i<numVec;i++) {
                oct->rrCoaxTransVec[i] = rrCoaxTransVec[i];
            }
            oct->numRRCoaxTransVec = numVec;
        } else {
            //allocate memory for temporary vector array
            tempVec = (float*)malloc((oct->numRRCoaxTransVec+numVec)*sizeof(float));
            
            //move all vectors in the array to the temporary array
            for(int i=0;i<oct->numRRCoaxTransVec;i++) {
                tempVec[i] = oct->rrCoaxTransVec[i];
            }
            
            //free the memory
            free(oct->rrCoaxTransVec);
            
            //tell one by one if the new vectors belong to the existing array
            for(int i=0;i<numVec;i++) {
                //assume that the new vector does not belong to the existing array
                flag = false;
                for(int j=0;j<oct->numRRCoaxTransVec;j++) {
                    if(abs(rrCoaxTransVec[i]-tempVec[j])<epsilon) {
                        //the current new vector is in the old array
                        flag = true;
                        break;
                    }
                }
                if(!flag) {
                    //push the vector into the vector array and increase the number of SR vectors
                    tempVec[oct->numRRCoaxTransVec++] = rrCoaxTransVec[i];
                }
            }
            
            //allocate memory for rr/ss translations
            oct->rrCoaxTransVec = (float*)malloc(oct->numRRCoaxTransVec*sizeof(float));
            for(int i=0;i<oct->numRRCoaxTransVec;i++) {
                oct->rrCoaxTransVec[i] = tempVec[i];
            }
            free(tempVec);
        }
        free(rrCoaxTransVec);
        free(ang);
    }
    
    bubbleSort(oct->rrCoaxTransVec,oct->numRRCoaxTransVec);
    
    //generate rr/ss coaxial translation matrices; the largest matrices corresponding to the current wave number will be generated
    if(oct->rrCoaxMat!=NULL) {
        free(oct->rrCoaxMat);
    }
    pmax = truncNum(wavNum,oct->eps,1.5,pow(2,-oct->lmin)*oct->d);
    oct->rrCoaxMat = (cuFloatComplex*)malloc(oct->numRRCoaxTransVec*(pmax*(2*pmax*pmax+3*pmax+1)/6)*sizeof(cuFloatComplex));
    HOST_CALL(genRRSparseCoaxTransMat(wavNum,oct->rrCoaxTransVec,oct->numRRCoaxTransVec,pmax,oct->rrCoaxMat));
    
    
    return EXIT_SUCCESS;
}

__host__ void destroyOctree(octree *oct)
{
    for(int l=oct->lmin;l<=oct->lmax;l++) {
        free(oct->fmmLevelSet[l-oct->lmin]);
    }
    free(oct->fmmLevelSet);
    
    free(oct->ang);
    free(oct->rotMat1);
    free(oct->rotMat2);
    
    free(oct->srCoaxTransVec);
    free(oct->srCoaxMat);
    
    free(oct->rrCoaxTransVec);
    free(oct->rrCoaxMat);
}

__host__ int OprL(const octree &oct, const cuFloatComplex *q, cuFloatComplex *prod)
{
    
}





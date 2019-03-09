/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <time.h>
#include <curand.h>
#include "fmm.h"


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

int genGaussParams(const int n, float *pt, float *wgt)
{
    int i, j;
    double t;
    gsl_vector *v = gsl_vector_alloc(n);
    for(i=0;i<n-1;i++) {
        gsl_vector_set(v,i,sqrt(pow(2*(i+1),2)-1));
    }
    for(i=0;i<n-1;i++) {
        t = gsl_vector_get(v,i);
        gsl_vector_set(v,i,(i+1)/t);
    }
    gsl_matrix *A = gsl_matrix_alloc(n,n);
    gsl_matrix *B = gsl_matrix_alloc(n,n);
    for(i=0;i<n;i++) {
        for(j=0;j<n;j++) {
            gsl_matrix_set(A,i,j,0);
            if(i==j) {
                gsl_matrix_set(B,i,j,1);
            } else {
                gsl_matrix_set(B,i,j,0);
            }
        }
    }
    for(i=0;i<n-1;i++) {
        t = gsl_vector_get(v,i);
        gsl_matrix_set(A,i+1,i,t);
        gsl_matrix_set(A,i,i+1,t);
    }
    gsl_eigen_symmv_workspace * wsp = gsl_eigen_symmv_alloc(n);
    HOST_CALL(gsl_eigen_symmv(A,v,B,wsp));
    for(i=0;i<n;i++) {
        pt[i] = gsl_vector_get(v,i);
        t = gsl_matrix_get(B,0,i);
        wgt[i] = 2*pow(t,2);
    }
    gsl_vector_free(v);
    gsl_matrix_free(A);
    gsl_matrix_free(B);
    return EXIT_SUCCESS;
}

__host__ __device__ cuFloatComplex cplxExp(const float theta){
    return make_cuFloatComplex(cos(theta),sin(theta));
}

__host__ __device__ float dotProd(const cartCoord u, const cartCoord v) {
    return u.x*v.x+u.y*v.y+u.z*v.z;
}

__host__ __device__ cartCoord crossProd(const cartCoord u, const cartCoord v) {
    cartCoord r;
    r.x = (u.y)*(v.z)-(u.z)*(v.y);
    r.y = (u.z)*(v.x)-(u.x)*(v.z);
    r.z = (u.x)*(v.y)-(u.y)*(v.x);
    return r;
}

__host__ __device__ cartCoord cartCoordAdd(const cartCoord u, const cartCoord v)
{
    cartCoord result;
    result.x = u.x+v.x;
    result.y = u.y+v.y;
    result.z = u.z+v.z;
    return result;
}

__host__ __device__ cartCoord cartCoordSub(const cartCoord u, const cartCoord v)
{
    cartCoord result;
    result.x = u.x-v.x;
    result.y = u.y-v.y;
    result.z = u.z-v.z;
    return result;
}

__host__ __device__ cartCoord scalarProd(const float lambda, const cartCoord v)
{
    cartCoord result;
    result.x = lambda*v.x;
    result.y = lambda*v.y;
    result.z = lambda*v.z;
    return result;
}

__host__ __device__ sphCoord cart2sph(const cartCoord s)
{
    sphCoord temp;
    temp.r = sqrtf(powf(s.x,2)+powf(s.y,2)+powf(s.z,2));
    temp.theta = acosf(s.z/(temp.r));
    temp.phi = atan2f(s.y,s.x);
    return temp;
}

__host__ __device__ cartCoord sph2cart(const sphCoord s)
{
    cartCoord temp;
    temp.x = s.r*sinf(s.theta)*cosf(s.phi);
    temp.y = s.r*sinf(s.theta)*sinf(s.phi);
    temp.z = s.r*cosf(s.theta);
    return temp;
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
        const int p, cuFloatComplex *mat) {
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

__device__ void coaxTransMatGen(cuFloatComplex *enlMat, const int p, cuFloatComplex *mat) 
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

__global__ void coaxTransMatsGen(cuFloatComplex *mats_enl, const int maxNum, const int p, 
        cuFloatComplex *mats)
{
    int idx_x = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx_x < maxNum) {
        coaxTransMatGen(&mats_enl[idx_x*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)],p,&mats[idx_x*p*p*p*p]);
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

int genSSCoaxTransMat(const float wavNum, const float *vec, const int numVec, const int p, 
        cuFloatComplex *mat)
{
    HOST_CALL(genRRCoaxTransMat(wavNum,vec,numVec,p,mat));
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

__device__ void rotMatGen(const rotAng rotAngle, const int p, float *H,  
        cuFloatComplex *rotMat)
{
    int np,n, mp, m, i, j, matIdx;
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

__global__ void rotMatsGen(const rotAng *rotAng, const int numRot, const int p, 
        float *H_enl,cuFloatComplex *rotMat)
{
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < numRot) {
        rotMatGen(rotAng[idx],p,&H_enl[idx*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)],&rotMat[idx*p*p*p*p]);
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

__global__ void cuRotsVecsMul(const cuFloatComplex *mat, const cuFloatComplex *vec, 
        const int num, const int p, cuFloatComplex *prod)
{
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < num) {
        int len = p*p;
        cuRotVecMul(&mat[idx*len*len],&vec[idx*len],p,&prod[idx*len]);
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







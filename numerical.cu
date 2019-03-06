/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */



#include "numerical.h"

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

__device__ void rrTransMatGen(cuFloatComplex *matPtr, cuFloatComplex *matPtr2, 
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

__global__ void rrTransMatsGen(cuFloatComplex *mats_enl,const int maxNum, 
        const int p, cuFloatComplex *mats)
{
    int idx_x = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx_x < maxNum) {
        rrTransMatGen(&mats_enl[idx_x*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)],&mats[idx_x*p*p*p*p],p);
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
    rrTransMatsGen<<<gridStruct,blockStruct>>>(enlMat_d,numVec,p,transMat_d);
    
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

__device__ void rrCoaxTransMatGen(cuFloatComplex *enlMat, const int p, cuFloatComplex *mat) 
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

__global__ void rrCoaxTransMatsGen(cuFloatComplex *mats_enl, const int maxNum, const int p, 
        cuFloatComplex *mats)
{
    int idx_x = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx_x < maxNum) {
        rrCoaxTransMatGen(&mats_enl[idx_x*(2*p-1)*(2*p-1)*(2*p-1)*(2*p-1)],p,&mats[idx_x*p*p*p*p]);
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
    
    rrCoaxTransMatsGen<<<gridStruct,blockStruct>>>(enlMat_d,numVec,p,mat_d);
    
    CUDA_CALL(cudaMemcpy(mat,mat_d,sizeof(cuFloatComplex)*p*p*p*p*numVec,cudaMemcpyDeviceToHost));
    
    CUDA_CALL(cudaFree(enlMat_d));
    CUDA_CALL(cudaFree(mat_d));
    free(enlMat_h);
    return EXIT_SUCCESS;
}
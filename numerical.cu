/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */



#include "numerical.h"

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


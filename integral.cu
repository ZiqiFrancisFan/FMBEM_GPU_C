/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "integral.h"
#include "translation.h"
#include "structs.h"

__constant__ float density = 1.2041;

__constant__ float speed = 343.21;

//Integral points and weights
__constant__ float INTPT[INTORDER]; 

__constant__ float INTWGT[INTORDER];

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

int gaussPtsToDevice(const float *pt, const float *wgt)
{
    CUDA_CALL(cudaMemcpyToSymbol(INTPT,pt,INTORDER*sizeof(float),0,cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(INTWGT,wgt,INTORDER*sizeof(float),0,cudaMemcpyHostToDevice));
    return EXIT_SUCCESS;
}

__host__ __device__ float cartNorm(const cartCoord x)
{
    return sqrtf(x.x*x.x+x.y*x.y+x.z*x.z);
}

__host__ __device__ float cartDist(const cartCoord x, const cartCoord y)
{
    return cartNorm(cartCoordSub(x,y));
}


__host__ __device__ cuFloatComplex green(const float k, const cartCoord x, const cartCoord y)
{
    float r = cartDist(x,y);
    cuFloatComplex numerator = cplxExp(k*r);
    float denomenator = 4*PI*r;
    return make_cuFloatComplex(cuCrealf(numerator)/denomenator,cuCimagf(numerator)/denomenator);
}

__host__ cuFloatComplex triElemIntegral_g_nsgl(const float wavNum, const cartCoord nod[3], const cartCoord y, 
        const float *pt, const float *wgt)
{
    float J = cartNorm(crossProd(cartCoordSub(nod[0],nod[2]),cartCoordSub(nod[1],nod[2])));
    float rho, theta, eta1, eta2, xi1, xi2, xi3, wn, wm, temp;
    cuFloatComplex g, result = make_cuFloatComplex(0,0);
    cartCoord x;
    for(int n=0;n<INTORDER;n++) {
        eta2 = pt[n];
        wn = wgt[n];
        theta = 0.5f+0.5f*eta2;
        for(int m=0;m<INTORDER;m++) {
            eta1 = pt[m];
            wm = wgt[m];
            rho = 0.5f+0.5f*eta1;
            temp = 0.25f*wn*wm*rho*J;
            xi1 = rho*(1-theta);
            xi2 = rho-xi1;
            xi3 = 1-xi1-xi2;
            x = {
                    nod[0].x*xi1+nod[1].x*xi2+nod[2].x*xi3,
                    nod[0].y*xi1+nod[1].y*xi2+nod[2].y*xi3,
                    nod[0].z*xi1+nod[1].z*xi2+nod[2].z*xi3
                };
            g = green(wavNum,x,y);
            result = cuCaddf(result,make_cuFloatComplex(temp*cuCrealf(g),temp*cuCimagf(g)));
        }
    }
    result = make_cuFloatComplex(0.25f*cuCrealf(result),0.25f*cuCimagf(result));
    return result;
}

__device__ cuFloatComplex triElemIntegral_g_nsgl(const float wavNum, const cartCoord nod[3], const cartCoord y)
{
    float J = cartNorm(crossProd(cartCoordSub(nod[0],nod[2]),cartCoordSub(nod[1],nod[2])));
    float rho, theta, eta1, eta2, xi1, xi2, xi3, wn, wm, temp;
    cuFloatComplex g, result = make_cuFloatComplex(0,0);
    cartCoord x;
    for(int n=0;n<INTORDER;n++) {
        eta2 = INTPT[n];
        wn = INTWGT[n];
        theta = 0.5f+0.5f*eta2;
        for(int m=0;m<INTORDER;m++) {
            eta1 = INTPT[m];
            wm = INTWGT[m];
            rho = 0.5f+0.5f*eta1;
            temp = 0.25f*wn*wm*rho*J;
            xi1 = rho*(1-theta);
            xi2 = rho-xi1;
            xi3 = 1-xi1-xi2;
            x = {
                    nod[0].x*xi1+nod[1].x*xi2+nod[2].x*xi3,
                    nod[0].y*xi1+nod[1].y*xi2+nod[2].y*xi3,
                    nod[0].z*xi1+nod[1].z*xi2+nod[2].z*xi3
                };
            g = green(wavNum,x,y);
            result = cuCaddf(result,make_cuFloatComplex(temp*cuCrealf(g),temp*cuCimagf(g)));
        }
    }
    result = make_cuFloatComplex(0.25f*cuCrealf(result),0.25f*cuCimagf(result));
    return result;
}






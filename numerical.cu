/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <gsl/gsl_sf.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>


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


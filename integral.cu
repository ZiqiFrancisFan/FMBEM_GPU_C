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

__host__ __device__ cuFloatComplex green(const float k, const cartCoord x, const cartCoord y)
{
    float r = cartDist(x,y);
    cuFloatComplex numerator = cplxExp(k*r);
    float denomenator = 4*PI*r;
    return make_cuFloatComplex(cuCrealf(numerator)/denomenator,cuCimagf(numerator)/denomenator);
}

__host__ __device__ cuFloatComplex triElemIntegral_G_nsgl(const float wavNum, const cartCoord nod[3], 
        const cartCoord y, const float *pt, const float *wgt)
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
    return result;
}

__device__ cuFloatComplex triElemIntegral_G_nsgl(const float wavNum, const cartCoord nod[3], const cartCoord y)
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
    return result;
}

__host__ __device__ cuFloatComplex triElemIntegral_G_sgl_3(const float wavNum, const cartCoord nod[3], 
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
            g = green(wavNum,x,nod[2]);
            result = cuCaddf(result,make_cuFloatComplex(temp*cuCrealf(g),temp*cuCimagf(g)));
        }
    }
    return result;
}

__device__ cuFloatComplex triElemIntegral_G_sgl_3(const float wavNum, const cartCoord nod[3])
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
            g = green(wavNum,x,nod[2]);
            result = cuCaddf(result,make_cuFloatComplex(temp*cuCrealf(g),temp*cuCimagf(g)));
        }
    }
    return result;
}

__host__ __device__ cuFloatComplex triElemIntegral_G_sgl(const float wavNum, const cartCoord nod[3], 
        const float *pt, const float *wgt)
{
    cartCoord y;
    cartCoord ctr23 = scalarMul(0.5,cartCoordAdd(nod[1],nod[2]));
    y = cartCoordAdd(nod[0],scalarMul(2.0f/3.0f,cartCoordSub(ctr23,nod[0])));
    cuFloatComplex result1 = make_cuFloatComplex(0,0), result2 = make_cuFloatComplex(0,0), 
            result3 = make_cuFloatComplex(0,0), result = make_cuFloatComplex(0,0);
    cartCoord nod_sub[3];
    nod_sub[0] = nod[0];
    nod_sub[1] = nod[1];
    nod_sub[2] = y;
    result1 = triElemIntegral_G_sgl_3(wavNum,nod_sub,pt,wgt);
    nod_sub[0] = nod[1];
    nod_sub[1] = nod[2];
    nod_sub[2] = y;
    result2 = triElemIntegral_G_sgl_3(wavNum,nod_sub,pt,wgt);
    nod_sub[0] = nod[2];
    nod_sub[1] = nod[0];
    nod_sub[2] = y;
    result3 = triElemIntegral_G_sgl_3(wavNum,nod_sub,pt,wgt);
    result = cuCaddf(cuCaddf(result1,result2),result3);
    return result;
}

__device__ cuFloatComplex triElemIntegral_G_sgl(const float wavNum, const cartCoord nod[3])
{
    cartCoord y;
    cartCoord ctr23 = scalarMul(0.5,cartCoordAdd(nod[1],nod[2]));
    y = cartCoordAdd(nod[0],scalarMul(2.0f/3.0f,cartCoordSub(ctr23,nod[0])));
    cuFloatComplex result1 = make_cuFloatComplex(0,0), result2 = make_cuFloatComplex(0,0), 
            result3 = make_cuFloatComplex(0,0), result = make_cuFloatComplex(0,0);
    cartCoord nod_sub[3];
    nod_sub[0] = nod[0];
    nod_sub[1] = nod[1];
    nod_sub[2] = y;
    result1 = triElemIntegral_G_sgl_3(wavNum,nod_sub);
    nod_sub[0] = nod[1];
    nod_sub[1] = nod[2];
    nod_sub[2] = y;
    result2 = triElemIntegral_G_sgl_3(wavNum,nod_sub);
    nod_sub[0] = nod[2];
    nod_sub[1] = nod[0];
    nod_sub[2] = y;
    result3 = triElemIntegral_G_sgl_3(wavNum,nod_sub);
    result = cuCaddf(cuCaddf(result1,result2),result3);
    return result;
}

__host__ cuFloatComplex triElemIntegral_R(const float wavNum, const cartCoord nod[3], 
        const int n, const int m, const cartCoord x_lp, const float *pt, const float *wgt)
{
    float J = cartNorm(crossProd(cartCoordSub(nod[0],nod[2]),cartCoordSub(nod[1],nod[2])));
    float rho, theta, eta1, eta2, xi1, xi2, xi3, wn, wm, temp;
    cuFloatComplex rgl, result = make_cuFloatComplex(0,0);
    cartCoord x;
    sphCoord sphTemp;
    for(int i=0;i<INTORDER;i++) {
        eta2 = pt[i];
        wn = wgt[i];
        theta = 0.5f+0.5f*eta2;
        for(int j=0;j<INTORDER;j++) {
            eta1 = pt[j];
            wm = wgt[j];
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
            sphTemp = cart2sph(cartCoordSub(x,x_lp));
            rgl = gsl_complex2cuFloatComplex(rglBasis(wavNum,n,m,sphTemp));
            result = cuCaddf(result,make_cuFloatComplex(temp*cuCrealf(rgl),temp*cuCimagf(rgl)));
        }
    }
    return result;
}

__host__ cuFloatComplex dirDev_R(const float wavNum, const int n, const int m, 
        const cartCoord nrml, const cartCoord x) 
{
    cuFloatComplex temp_c[6];
    float temp_f[6];
    sphCoord coord_sph = cart2sph(x);
    temp_c[0] = gsl_complex2cuFloatComplex(rglBasis(wavNum,n+1,m+1,coord_sph));
    if(n-1<0) {
        temp_c[1] = make_cuFloatComplex(0,0);
    } else {
        temp_c[1] = gsl_complex2cuFloatComplex(rglBasis(wavNum,n-1,m+1,coord_sph));
    }
    temp_c[2] = gsl_complex2cuFloatComplex(rglBasis(wavNum,n+1,m-1,coord_sph));
    if(n-1<0) {
        temp_c[3] = make_cuFloatComplex(0,0);
    } else {
        temp_c[3] = gsl_complex2cuFloatComplex(rglBasis(wavNum,n-1,m-1,coord_sph));
    }
    if(n-1<0) {
        temp_c[4] = make_cuFloatComplex(0,0);
    } else {
        temp_c[4] = gsl_complex2cuFloatComplex(rglBasis(wavNum,n-1,m,coord_sph));
    } 
    temp_c[5] = gsl_complex2cuFloatComplex(rglBasis(wavNum,n+1,m,coord_sph));
    temp_f[0] = bCoeff(n+1,-(m+1));
    temp_f[1] = bCoeff(n,m);
    temp_f[2] = bCoeff(n+1,m-1);
    temp_f[3] = bCoeff(n,-m);
    temp_f[4] = aCoeff(n-1,m);
    temp_f[5] = aCoeff(n,m);
    temp_c[0] = make_cuFloatComplex(cuCrealf(temp_c[0])*temp_f[0],cuCimagf(temp_c[0])*temp_f[0]);
    temp_c[1] = make_cuFloatComplex(cuCrealf(temp_c[1])*temp_f[1],cuCimagf(temp_c[1])*temp_f[1]);
    temp_c[2] = make_cuFloatComplex(cuCrealf(temp_c[2])*temp_f[2],cuCimagf(temp_c[2])*temp_f[2]);
    temp_c[3] = make_cuFloatComplex(cuCrealf(temp_c[3])*temp_f[3],cuCimagf(temp_c[3])*temp_f[3]);
    temp_c[4] = make_cuFloatComplex(cuCrealf(temp_c[4])*temp_f[4],cuCimagf(temp_c[4])*temp_f[4]);
    temp_c[5] = make_cuFloatComplex(cuCrealf(temp_c[5])*temp_f[5],cuCimagf(temp_c[5])*temp_f[5]);
    temp_c[0] = cuCsubf(temp_c[0],temp_c[1]);
    temp_c[1] = cuCsubf(temp_c[2],temp_c[3]);
    temp_c[2] = cuCsubf(temp_c[4],temp_c[5]);
    temp_c[3] = make_cuFloatComplex(wavNum/2.0f*nrml.x,-wavNum/2.0f*nrml.y);
    temp_c[4] = make_cuFloatComplex(wavNum/2.0f*nrml.x,wavNum/2.0f*nrml.y);
    temp_c[0] = cuCmulf(temp_c[3],temp_c[0]);
    temp_c[1] = cuCmulf(temp_c[4],temp_c[1]);
    temp_c[2] = make_cuFloatComplex(wavNum*nrml.z*cuCrealf(temp_c[2]),wavNum*nrml.z*cuCimagf(temp_c[2]));
    temp_c[0] = cuCaddf(temp_c[0],temp_c[1]);
    temp_c[0] = cuCaddf(temp_c[0],temp_c[2]);
    return temp_c[0];
}

__host__ cuFloatComplex triElemIntegral_pRpn(const float wavNum, const cartCoord nod[3], 
        const int n, const int m, const cartCoord x_lp, const float *pt, const float *wgt)
{
    cartCoord nrml = crossProd(cartCoordSub(nod[0],nod[2]),cartCoordSub(nod[1],nod[2]));
    cartCoord nrml_nrmlzd = normalize(nrml);
    float J = cartNorm(nrml);
    float rho, theta, eta1, eta2, xi1, xi2, xi3, wn, wm, temp;
    cuFloatComplex pRpn, result = make_cuFloatComplex(0,0);
    cartCoord x;
    for(int i=0;i<INTORDER;i++) {
        eta2 = pt[i];
        wn = wgt[i];
        theta = 0.5f+0.5f*eta2;
        for(int j=0;j<INTORDER;j++) {
            eta1 = pt[j];
            wm = wgt[j];
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
            x = cartCoordSub(x,x_lp);
            pRpn = dirDev_R(wavNum,n,m,nrml_nrmlzd,x);
            result = cuCaddf(result,make_cuFloatComplex(temp*cuCrealf(pRpn),temp*cuCimagf(pRpn)));
        }
    }
    return result;
}

__host__ __device__ float prpn1(const cartCoord n, const cartCoord x, const cartCoord y)
{
    float dist = cartDist(x,y);
    return ((x.x-y.x)*n.x+(x.y-y.y)*n.y+(x.z-y.z)*n.z)/dist;
}

__host__ __device__ float prpn2(const cartCoord n, const cartCoord x, const cartCoord y)
{
    float dist = cartDist(x,y);
    return ((y.x-x.x)*n.x+(y.y-x.y)*n.y+(y.z-x.z)*n.z)/dist;
}

__host__ __device__ float prRecippn1(const cartCoord n, const cartCoord x, const cartCoord y)
{
    float dist = cartDist(x,y);
    return -1.0f/(dist*dist)*prpn1(n,x,y);
}

__host__ __device__ float prRecippn2(const cartCoord n, const cartCoord x, const cartCoord y)
{
    float dist = cartDist(x,y);
    return -1.0f/(dist*dist)*prpn2(n,x,y);
}

__host__ __device__ cuFloatComplex pGp1n(const float wavNum, const cartCoord x, const cartCoord y, 
        const cartCoord n)
{
    cuFloatComplex temp_c[2];
    float temp_f, dist = cartDist(x,y);
    temp_c[0] = green(wavNum,x,y);
    temp_c[1] = make_cuFloatComplex(-1.0f/dist,wavNum);
    temp_f = prpn1(n,x,y);
    temp_c[0] = cuCmulf(temp_c[0],temp_c[1]);
    temp_c[0] = make_cuFloatComplex(cuCrealf(temp_c[0])*temp_f,cuCimagf(temp_c[0])*temp_f);
    return temp_c[0];
}

__host__ __device__ cuFloatComplex pGp2n(const float wavNum, const cartCoord n, 
        const cartCoord x, const cartCoord y)
{
    cuFloatComplex temp_c[2];
    float temp_f, dist = cartDist(x,y);
    temp_c[0] = green(wavNum,x,y);
    temp_c[1] = make_cuFloatComplex(-1.0f/dist,wavNum);
    temp_f = prpn2(n,x,y);
    temp_c[0] = cuCmulf(temp_c[0],temp_c[1]);
    temp_c[0] = make_cuFloatComplex(cuCrealf(temp_c[0])*temp_f,cuCimagf(temp_c[0])*temp_f);
    return temp_c[0];
}

__host__ __device__ cuFloatComplex triElemIntegral_pGp1n_nsgl(const float wavNum, 
        const cartCoord nod[3], const cartCoord y, const float *pt, const float *wgt)
{
    cartCoord nrml = crossProd(cartCoordSub(nod[0],nod[2]),cartCoordSub(nod[1],nod[2]));
    cartCoord nrml_nrmlzd = normalize(nrml);
    float J = cartNorm(nrml);
    float rho, theta, eta1, eta2, xi1, xi2, xi3, wn, wm, temp;
    cuFloatComplex pgpn1, result = make_cuFloatComplex(0,0);
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
            pgpn1 = pGp1n(wavNum,x,y,nrml_nrmlzd);
            result = cuCaddf(result,make_cuFloatComplex(temp*cuCrealf(pgpn1),
                    temp*cuCimagf(pgpn1)));
        }
    }
    return result;
}

__host__ __device__ cuFloatComplex triElemIntegral_pGp1n_sgl_3(const float wavNum, const cartCoord nod[3], 
        const float *pt, const float *wgt)
{
    cartCoord nrml = crossProd(cartCoordSub(nod[0],nod[2]),cartCoordSub(nod[1],nod[2]));
    cartCoord nrml_nrmlzd = normalize(nrml);
    float J = cartNorm(nrml);
    float rho, theta, eta1, eta2, xi1, xi2, xi3, wn, wm, temp;
    cuFloatComplex pgpn1, result = make_cuFloatComplex(0,0);
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
            pgpn1 = pGp1n(wavNum,x,nod[2],nrml_nrmlzd);
            result = cuCaddf(result,make_cuFloatComplex(temp*cuCrealf(pgpn1),
                    temp*cuCimagf(pgpn1)));
        }
    }
    return result;
}

__host__ __device__ cuFloatComplex triElemIntegral_pGp1n_sgl(const float wavNum, const cartCoord nod[3], 
        const float *pt, const float *wgt)
{
    cartCoord y;
    cartCoord ctr23 = scalarMul(0.5,cartCoordAdd(nod[1],nod[2]));
    y = cartCoordAdd(nod[0],scalarMul(2.0f/3.0f,cartCoordSub(ctr23,nod[0])));
    cuFloatComplex result1 = make_cuFloatComplex(0,0), result2 = make_cuFloatComplex(0,0), 
            result3 = make_cuFloatComplex(0,0), result = make_cuFloatComplex(0,0);
    cartCoord nod_sub[3];
    nod_sub[0] = nod[0];
    nod_sub[1] = nod[1];
    nod_sub[2] = y;
    result1 = triElemIntegral_pGp1n_sgl_3(wavNum,nod_sub,pt,wgt);
    nod_sub[0] = nod[1];
    nod_sub[1] = nod[2];
    nod_sub[2] = y;
    result2 = triElemIntegral_pGp1n_sgl_3(wavNum,nod_sub,pt,wgt);
    nod_sub[0] = nod[2];
    nod_sub[1] = nod[0];
    nod_sub[2] = y;
    result3 = triElemIntegral_pGp1n_sgl_3(wavNum,nod_sub,pt,wgt);
    result = cuCaddf(cuCaddf(result1,result2),result3);
    return result;
}

__host__ __device__ cuFloatComplex triElemIntegral_pGp2n_nsgl(const float wavNum, 
        const cartCoord nod[3], const cartCoord nrml, const cartCoord y, 
        const float *pt, const float *wgt)
{
    float J = cartNorm(crossProd(cartCoordSub(nod[0],nod[2]),cartCoordSub(nod[1],nod[2])));
    float rho, theta, eta1, eta2, xi1, xi2, xi3, wn, wm, temp;
    cuFloatComplex pgpn2, result = make_cuFloatComplex(0,0);
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
            pgpn2 = pGp2n(wavNum,nrml,x,y);
            result = cuCaddf(result,make_cuFloatComplex(temp*cuCrealf(pgpn2),
                    temp*cuCimagf(pgpn2)));
        }
    }
    return result;
}

__host__ __device__ cuFloatComplex triElemIntegral_pGp2n_sgl_3(const float wavNum, 
        const cartCoord nod[3], const float *pt, const float *wgt)
{
    cartCoord nrml = crossProd(cartCoordSub(nod[0],nod[2]),cartCoordSub(nod[1],nod[2]));
    cartCoord nrml_nrmlzd = normalize(nrml);
    float J = cartNorm(nrml);
    float rho, theta, eta1, eta2, xi1, xi2, xi3, wn, wm, temp;
    cuFloatComplex pgpn2, result = make_cuFloatComplex(0,0);
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
            pgpn2 = pGp2n(wavNum,nrml_nrmlzd,x,nod[2]);
            result = cuCaddf(result,make_cuFloatComplex(temp*cuCrealf(pgpn2),
                    temp*cuCimagf(pgpn2)));
        }
    }
    return result;
}

__host__ __device__ cuFloatComplex triElemIntegral_pGp2n_sgl(const float wavNum, 
        const cartCoord nod[3], const float *pt, const float *wgt)
{
    cartCoord y;
    cartCoord ctr23 = scalarMul(0.5,cartCoordAdd(nod[1],nod[2]));
    y = cartCoordAdd(nod[0],scalarMul(2.0f/3.0f,cartCoordSub(ctr23,nod[0])));
    cuFloatComplex result1 = make_cuFloatComplex(0,0), result2 = make_cuFloatComplex(0,0), 
            result3 = make_cuFloatComplex(0,0), result = make_cuFloatComplex(0,0);
    cartCoord nod_sub[3];
    nod_sub[0] = nod[0];
    nod_sub[1] = nod[1];
    nod_sub[2] = y;
    result1 = triElemIntegral_pGp2n_sgl_3(wavNum,nod_sub,pt,wgt);
    nod_sub[0] = nod[1];
    nod_sub[1] = nod[2];
    nod_sub[2] = y;
    result2 = triElemIntegral_pGp2n_sgl_3(wavNum,nod_sub,pt,wgt);
    nod_sub[0] = nod[2];
    nod_sub[1] = nod[0];
    nod_sub[2] = y;
    result3 = triElemIntegral_pGp2n_sgl_3(wavNum,nod_sub,pt,wgt);
    result = cuCaddf(cuCaddf(result1,result2),result3);
    return result;
}

__host__ __device__ cuFloatComplex p2Gp1np2n(const float wavNum, const cartCoord n1, const cartCoord n2, 
        const cartCoord x, const cartCoord y)
{
    cuFloatComplex temp[3];
    float dist, t;
    
    dist = cartDist(x,y);
    temp[0] = cplxExp(wavNum*dist);
    t = 4*PI*dist*dist*dist;
    temp[0] = make_cuFloatComplex(cuCrealf(temp[0])/t,cuCimagf(temp[0])/t);
    
    temp[1] = make_cuFloatComplex(3-wavNum*wavNum*dist*dist,-3*wavNum*dist);
    t = prpn1(n1,x,y)*prpn2(n2,x,y);
    temp[1] = make_cuFloatComplex(cuCrealf(temp[1])*t,cuCimagf(temp[1])*t);
    
    t = dotProd(n1,n2);
    temp[2] = make_cuFloatComplex(1,-wavNum*dist);
    temp[2] = make_cuFloatComplex(cuCrealf(temp[2])*t,cuCimagf(temp[2])*t);
    
    return cuCmulf(temp[0],cuCaddf(temp[1],temp[2]));
}

__host__ __device__ cuFloatComplex triElemIntegral_p2Gp1np2n_nsgl(const float wavNum, 
        const cartCoord nod[3], const cartCoord nrml_y, const cartCoord y, 
        const float *pt, const float *wgt)
{
    cartCoord nrml_x = crossProd(cartCoordSub(nod[0],nod[2]),cartCoordSub(nod[1],nod[2]));
    cartCoord nrml_x_nrmlzd = normalize(nrml_x);
    float J = cartNorm(nrml_x);
    float rho, theta, eta1, eta2, xi1, xi2, xi3, wn, wm, temp;
    cuFloatComplex p2gpn1pn2, result = make_cuFloatComplex(0,0);
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
            p2gpn1pn2 = p2Gp1np2n(wavNum,nrml_x_nrmlzd,nrml_y,x,y);
            result = cuCaddf(result,make_cuFloatComplex(temp*cuCrealf(p2gpn1pn2),
                    temp*cuCimagf(p2gpn1pn2)));
        }
    }
    return result;
}

__host__ __device__ cartCoord triCentroid(const cartCoord nod[3])
{
    cartCoord ctr_23 = scalarMul(0.5,cartCoordAdd(nod[1],nod[2]));
    cartCoord centroid = cartCoordAdd(nod[0],scalarMul(2.0/3.0,cartCoordSub(ctr_23,nod[0])));
    return centroid;
}

__host__ __device__ float theta2rho(const cartCoord nod[3], const float theta) 
{
    cartCoord vc = triCentroid(nod);
    float t, rho;
    //c1 is the polar axis
    float d_c1 = cartDist(vc,nod[0]); //distance between center and node 1
    float d_c2 = cartDist(vc,nod[1]); //distance between center and node 2
    float d_c3 = cartDist(vc,nod[2]); //distance between center and node 3
    float d_12 = cartDist(nod[0],nod[1]); //distance between node 1 and node 2
    float d_23 = cartDist(nod[1],nod[2]); //distance between node 2 and node 3
    float d_31 = cartDist(nod[2],nod[0]); //distance between node 3 and node 1
    //float theta_1 = 0;
    t = (powf(d_c1,2)+powf(d_c2,2)-powf(d_12,2))/(2*d_c1*d_c2);
    float theta_2 = acosf(t); //angle 1c2
    t = (powf(d_c2,2)+powf(d_c3,2)-powf(d_23,2))/(2*d_c2*d_c3);
    float theta_3 = theta_2+acosf(t); //angle 1c3 in the order of 1c2 and 2c3
    //coordinate of node 1 in the local coordinate system
    float x_1 = d_c1;
    float y_1 = 0;
    //coordinate of node 2 in the local coordinate system
    float x_2 = d_c2*cosf(theta_2);
    float y_2 = d_c2*sinf(theta_2);
    //coordinate of node 3 in the local coordinate system
    float x_3 = d_c3*cosf(theta_3);
    float y_3 = d_c3*sinf(theta_3);
    float x_i, y_i; //intersection point
    float k; //slope
    
    if(theta<theta_2) {
        //the intersection point is with line 12
        if(theta==0.5*PI) {
            x_i = 0;
            y_i = (y_2-y_1)/(x_2-x_1)*(-x_1)+y_1;
        } else {
            k = tanf(theta);
            x_i = (y_1-(y_2-y_1)/(x_2-x_1)*x_1)/(k-(y_2-y_1)/(x_2-x_1));
            y_i = k*x_i;
        }
    }
    if(theta>=theta_2 && theta<theta_3) {
        //intersects with line 32
        if(theta==0.5*PI) {
            x_i = 0;
            y_i = (y_3-y_2)/(x_3-x_2)*(-x_2)+y_2;
        } else {
            k = tan(theta);
            x_i = (y_2-(y_3-y_2)/(x_3-x_2)*x_2)/(k-(y_3-y_2)/(x_3-x_2));
            y_i = k*x_i;
        }
    }
    if(theta>=theta_3) {
        //intersects with line 21
        if(theta==0.5*PI) {
            x_i = 0;
            y_i = (y_1-y_3)/(x_1-x_3)*(-x_3)+y_3;
        } else {
            k = tan(theta);
            x_i = (y_3-(y_1-y_3)/(x_1-x_3)*x_3)/(k-(y_1-y_3)/(x_1-x_3));
            y_i = k*x_i;
        }
    }
    rho = sqrtf(powf(x_i,2)+powf(y_i,2));
    return rho;
}

__host__ __device__ cuFloatComplex triElemIntegral_p2Gp1np2n_sgl(const float wavNum, 
        const cartCoord nod[3], const float *pt, const float *wgt)
{
    float theta, rho, s, w;
    cuFloatComplex sum, t;
    sum = make_cuFloatComplex(0,0);
    for(int i=0;i<INTORDER;i++) {
        s = pt[i];
        w = wgt[i];
        theta = PI*s+PI;
        rho = theta2rho(nod,theta);
        t = make_cuFloatComplex(cosf(wavNum*rho),sinf(wavNum*rho));
        t = make_cuFloatComplex(1.0f/rho*cuCrealf(t),1.0f/rho*cuCimagf(t));
        t = make_cuFloatComplex(w*cuCrealf(t),w*cuCimagf(t));
        sum = cuCaddf(sum,t);
    }
    sum = make_cuFloatComplex(PI*cuCrealf(sum),PI*cuCimagf(sum));
    sum = cuCsubf(sum,make_cuFloatComplex(0,2*PI*wavNum));
    sum = make_cuFloatComplex(-1.0f/(4*PI)*cuCrealf(sum),-1.0f/(4*PI)*cuCimagf(sum));
    return sum;
}

__host__ __device__ void cmptDiffCoeff(const float wavNum, const cuFloatComplex *coeff, 
        const int p, const cartCoord nrml, cuFloatComplex *coeff_n)
{
    cuFloatComplex c[6], temp_c[6];
    float temp_f;
    for(int n=0;n<p;n++) {
        for(int m=-n;m<=n;m++) {
            if(abs(m-1)>n-1) {
                c[0] = make_cuFloatComplex(0,0);
            } else {
                c[0] = coeff[NM2IDX0(n-1,m-1)];
            }
            if(abs(m-1)>n+1) {
                c[1] = make_cuFloatComplex(0,0);
            } else {
                c[1] = coeff[NM2IDX0(n+1,m-1)];
            }
            if(abs(m+1)>n-1) {
                c[2] = make_cuFloatComplex(0,0);
            } else {
                c[2] = coeff[NM2IDX0(n-1,m+1)];
            }
            if(abs(m+1)>n+1) {
                c[3] = make_cuFloatComplex(0,0);
            } else {
                c[3] = coeff[NM2IDX0(n+1,m+1)];
            }
            if(abs(m)>n+1) {
                c[4] = make_cuFloatComplex(0,0);
            } else {
                c[4] = coeff[NM2IDX0(n+1,m)];
            }
            if(abs(m)>n-1) {
                c[5] = make_cuFloatComplex(0,0);
            } else {
                c[5] = coeff[NM2IDX0(n-1,m)];
            }
            temp_f = bCoeff(n,-m);
            temp_c[0] = make_cuFloatComplex(temp_f*cuCrealf(c[0]),temp_f*cuCimagf(c[0]));
            temp_f = bCoeff(n+1,m-1);
            temp_c[1] = make_cuFloatComplex(temp_f*cuCrealf(c[1]),temp_f*cuCimagf(c[1]));
            temp_f = bCoeff(n,m);
            temp_c[2] = make_cuFloatComplex(temp_f*cuCrealf(c[2]),temp_f*cuCimagf(c[2]));
            temp_f = bCoeff(n+1,-m-1);
            temp_c[2] = make_cuFloatComplex(temp_f*cuCrealf(c[3]),temp_f*cuCimagf(c[3]));
            temp_f = aCoeff(n,m);
            temp_c[2] = make_cuFloatComplex(temp_f*cuCrealf(c[4]),temp_f*cuCimagf(c[4]));
            temp_f = aCoeff(n-1,m);
            temp_c[2] = make_cuFloatComplex(temp_f*cuCrealf(c[5]),temp_f*cuCimagf(c[5]));
            
            temp_c[0] = cuCsubf(temp_c[0],temp_c[1]);
            temp_c[1] = cuCsubf(temp_c[2],temp_c[3]);
            temp_c[2] = cuCsubf(temp_c[4],temp_c[5]);
            
            temp_f = wavNum/2;
            temp_c[3] = make_cuFloatComplex(temp_f*nrml.x,-temp_f*nrml.y);
            temp_c[4] = make_cuFloatComplex(temp_f*nrml.x,temp_f*nrml.y);
            
            temp_c[0] = cuCmulf(temp_c[0],temp_c[3]);
            temp_c[1] = cuCmulf(temp_c[1],temp_c[4]);
            temp_f = wavNum*nrml.z;
            temp_c[2] = make_cuFloatComplex(temp_f*cuCrealf(temp_c[2]),temp_f*cuCimagf(temp_c[2]));
            
            coeff_n[NM2IDX0(n,m)] = cuCaddf(cuCaddf(temp_c[0],temp_c[1]),temp_c[2]);
        }
    }
}








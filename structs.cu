/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "structs.h"

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

__host__ __device__ cartCoord scalarMul(const float lambda, const cartCoord v)
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

__host__ __device__ float cartNorm(const cartCoord x)
{
    return sqrtf(x.x*x.x+x.y*x.y+x.z*x.z);
}

__host__ __device__ float cartDist(const cartCoord x, const cartCoord y)
{
    return cartNorm(cartCoordSub(x,y));
}

__host__ __device__ cartCoord normalize(const cartCoord x)
{
    return scalarMul(1.0f/cartNorm(x),x);
}

__host__ __device__ cartCoord_d cartCoordAdd_d(const cartCoord_d u, const cartCoord_d v)
{
    cartCoord_d result;
    result.x = u.x+v.x;
    result.y = u.y+v.y;
    result.z = u.z+v.z;
    return result;
}

__host__ __device__ cartCoord_d cartCoordSub_d(const cartCoord_d u, const cartCoord_d v)
{
    cartCoord_d result;
    result.x = u.x-v.x;
    result.y = u.y-v.y;
    result.z = u.z-v.z;
    return result;
}

__host__ __device__ cartCoord_d scalarMul_d(const double lambda, const cartCoord_d v)
{
    cartCoord_d result;
    result.x = lambda*v.x;
    result.y = lambda*v.y;
    result.z = lambda*v.z;
    return result;
}

__host__ __device__ cartCoord_d triCentroid_d(const cartCoord_d nod[3])
{
    cartCoord_d ctr_23 = scalarMul_d(0.5,cartCoordAdd_d(nod[1],nod[2]));
    cartCoord_d centroid = cartCoordAdd_d(nod[0],scalarMul_d(2.0/3.0,cartCoordSub_d(ctr_23,nod[0])));
    return centroid;
}


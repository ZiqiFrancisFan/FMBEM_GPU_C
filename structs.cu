/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "structs.h"
#include <stdio.h>

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

__host__ __device__ sphCoord_d cart_d2sph_d(const cartCoord_d s)
{
    sphCoord_d temp;
    temp.r = sqrt(pow(s.x,2)+pow(s.y,2)+pow(s.z,2));
    temp.theta = acos(s.z/(temp.r));
    temp.phi = atan2(s.y,s.x);
    return temp;
}

__host__ __device__ sphCoord sphCoord_d2sphCoord(const sphCoord_d s)
{
    sphCoord temp;
    temp.r = s.r;
    temp.theta = s.theta;
    temp.phi = s.phi;
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

__host__ __device__ cartCoord cartCoord_d2cartCoord(const cartCoord_d s)
{
    cartCoord t;
    t.x = s.x;
    t.y = s.y;
    t.z = s.z;
    return t;
}

__host__ __device__ cartCoord_d cartCoord2cartCoord_d(const cartCoord s)
{
    cartCoord_d temp;
    temp.x = s.x;
    temp.y = s.y;
    temp.z = s.z;
    return temp;
}

__host__ void printRotAngArray(const rotAng *angle, const int numAng)
{
    for(int i=0;i<numAng;i++) {
        printf("(%f,%f,%f)\n",angle[i].alpha,angle[i].beta,
                angle[i].gamma);
    }
}

__host__ void printCartCoordArray_d(const cartCoord_d *arr, const int num)
{
    for(int i=0;i<num;i++) {
        printf("(%f,%f,%f)\n",arr[i].x,arr[i].y,arr[i].z);
    }
}

__host__ void printCartCoordArray(const cartCoord *arr, const int num)
{
    for(int i=0;i<num;i++) {
        printf("(%f,%f,%f)\n",arr[i].x,arr[i].y,arr[i].z);
    }
}

__host__ void printSphCoordArray(const sphCoord *arr, const int num)
{
    for(int i=0;i<num;i++) {
        printf("(%f,%f,%f)\n",arr[i].r,arr[i].theta,arr[i].phi);
    }
}

__host__ void printFloatArray(const float *arr, const int num)
{
    for(int i=0;i<num;i++) {
        printf("%f\n",arr[i]);
    }
}

__host__ void printIntArray(const int *arr, const int num)
{
    for(int i=0;i<num;i++) {
        printf("%d\n",arr[i]);
    }
}

__host__ bool equalRotArrays(const rotAng *ang1, const rotAng *ang2, const int numAng)
{
    bool result = true, tempResult;
    for(int i=0;i<numAng;i++) {
        //assume that ang1[i] is not in ang2
        tempResult = false;
        for(int j=0;j<numAng;j++) {
            if(ang1[i].alpha==ang2[j].alpha && ang1[i].beta==ang2[j].beta && ang1[i].gamma==ang2[j].gamma) {
                tempResult = true;
                break;
            }
        }
        if(!tempResult) {
            result = false;
            break;
        }
    }
    return result;
}

__host__ void swap(float *a, float *b)
{
    float t = *a;
    *a = *b;
    *b = t;
}

__host__ void swap(rotAng *a, rotAng *b)
{
    rotAng temp = *a;
    *a = *b;
    *b = temp;
}

__host__ void bubbleSort(float *arr, const int n)
{ 
   int i, j; 
   for (i = 0; i < n-1; i++) {
       // Last i elements are already in place    
       for (j = 0; j < n-i-1; j++) {
           if(arr[j] > arr[j+1]) {
               swap(&arr[j], &arr[j+1]); 
           }
       }
   }
}

__host__ void sortRotArray(rotAng *ang, const int numRot, const float eps)
{
    int i, j;
    
    //sort the first angle
    for(i=0;i<numRot;i++) {
        for(j=0;j<numRot-i-1;j++) {
            if(ang[j].alpha > ang[j+1].alpha) {
                swap(&ang[j],&ang[j+1]);
            }
        }
    }
    
    //sort the second angle
    int idx = 0, num = 0;
    float alpha = ang[idx].alpha;
    
    //idx+num is the total number of rotations
    while(idx+num < numRot) {
        //still the same alpha
        if(abs(ang[idx+num].alpha-alpha)<eps) {
            num++;
        } else {
            //sort the current alpha segment
            for(i=0;i<num;i++) {
                for(j=idx;j<idx+num-i-1;j++) {
                    if(ang[j].beta > ang[j+1].beta) {
                        swap(&ang[j],&ang[j+1]);
                    }
                }
            }
            //update the new starting index
            alpha = ang[idx+num].alpha;
            idx += num;
            num = 0;
        }
    }
    
    for(i=0;i<num;i++) {
        for(j=idx;j<idx+num-i-1;j++) {
            if(ang[j].beta > ang[j+1].beta) {
                swap(&ang[j],&ang[j+1]);
            }
        }
    }
}

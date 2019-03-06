/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuComplex.h>
#include "octree.h"
#include "numerical.h"

/*We follow the rule that the small index of an array corresponds to the more 
 significant bit*/

void printIntArray(int *a, int sz) {
    int i;
    for(i=0;i<sz;i++) {
        printf("%d ",a[i]);
    }
    printf("\n");
}

int arrEqual(const int *a, const int *b, const int num) 
{
    int i;
    int result = 1;
    for(i=0;i<num;i++) {
        if(a[i]!=b[i]) {
            result = 0;
            break;
        }
    }
    return result;
}

void printSet(const int *set) 
{
    int i;
    if(set[0]==0) {
        printf("The set is empty.");
    } else {
        for(i=1;i<=set[0];i++) {
            printf("%d ",set[i]);
        }
    }
    printf("\n");
}

int parent(int num) 
{
    return num/8;
}

int child(int num, int cld) 
{
    return 8*num+cld;
}

int children(const int num, int *cldrn) 
{
    int i;
    for(i=0;i<8;i++) {
        cldrn[i] = child(num,i);
    }
} 

cartCoord_d scale(const cartCoord_d x, const cartCoord_d x_min, const double d) 
{
    cartCoord_d x_scaled;
    x_scaled.x = (x.x-x_min.x)/d;
    x_scaled.y = (x.y-x_min.y)/d;
    x_scaled.z = (x.z-x_min.z)/d;
    return x_scaled;
}

cartCoord_d descale(const cartCoord_d x_s, const cartCoord_d x_min, const double d)
{
    cartCoord_d x_ds;
    x_ds.x = d*x_s.x+x_min.x;
    x_ds.y = d*x_s.y+x_min.y;
    x_ds.z = d*x_s.z+x_min.z;
    return x_ds;
}

void scalePnts(const cartCoord_d *pnt, const int numPnts, const cartCoord_d pnt_min, 
        const double d, cartCoord_d* pnt_scaled)
{
    int i;
    for(i=0;i<numPnts;i++) {
        pnt_scaled[i] = scale(pnt[i],pnt_min,d);
    }
}

void dec2bin_frac(double s, int l, int *h)
{
    //the decimal number s is normalized to (0,1);
    //left is least significant while right is most significant
    if(s>=1 || s<=0) {
        //printf("The number is out of range in dec2bin_frac.\n");
    }
    int i;
    double t;
    for(i=1;i<=l;i++) {
        s*=2;
        t = floor(s);
        h[i-1] = (int)t;
        s-=t;
    }
    /*
    printf("0.");
    for(i=1;i<=l;i++) {
        printf("%d",h[i-1]);
    }
    printf("\n");
     */ 
}

void dec2bin_int(unsigned num, int *rep, int *numBits)
{
    //it is assumed that rep has size NUM_BITs_BIN
    int i = 0, j = 0;
    int temp;
    if(num == 0) {
        *numBits = 1;
        rep[0] = 0;
    } else {
        while(num!=0) {
            rep[i++] = num%2;
            num/=2;
        }
        *numBits = i; 
        for(j=0;j<i/2;j++) {
            temp = rep[j];
            rep[j] = rep[i-1-j];
            rep[i-1-j] = temp;
        }
    }
}

void bitIntleave(const int *x, const int *y, const int *z, const int l, int *result)
{
    //output is an oct array
    int i;
    for(i=1;i<=l;i++) {
        result[i-1] = pow(2,2)*x[i-1]+pow(2,1)*y[i-1]+pow(2,0)*z[i-1];
    }
}

int indArr2num(const int *ind, const int l, const int d)
{
    int i;
    int result = 0;
    for(i=0;i<l;i++) 
    {
        result+=pow(pow(2,d),l-1-i)*ind[i];
    }
    return result;
}

void bitDeintleave(const int *result, const int l, int *x, int *y, int *z)
{
    //result is an array containing 3*l elements of binary number
    int i;
    for(i=0;i<l;i++) 
    {
        x[i] = result[3*i];
        y[i] = result[3*i+1];
        z[i] = result[3*i+2];
    }
}

int pnt2boxnum(const cartCoord_d pnt, const int l)
{
    int *ind_x, *ind_y, *ind_z, *ind;
    int result;
    ind_x = (int*)malloc(l*sizeof(int));
    ind_y = (int*)malloc(l*sizeof(int));
    ind_z = (int*)malloc(l*sizeof(int));
    ind = (int*)malloc(l*sizeof(int));
    
    dec2bin_frac(pnt.x,l,ind_x);
    dec2bin_frac(pnt.y,l,ind_y);
    dec2bin_frac(pnt.z,l,ind_z);
    bitIntleave(ind_x,ind_y,ind_z,l,ind);

    result = indArr2num(ind,l,3);
    free(ind_x);
    free(ind_y);
    free(ind_z);
    free(ind);
    return result;
}

cartCoord_d boxCenter(const int num, const int l) {
    if(num < 0 || num > pow(pow(2,3),l)-1) {
        printf("Illegal input.\n");
        printf("Error at %s:%d\n",__FILE__,__LINE__);
        cartCoord_d temp = {nan("1"),nan("2"),nan("3")};
        return temp;
    }
    cartCoord_d pnt;
    int i;
    int *ind = (int*)malloc(3*l*sizeof(int));
    int *rep = (int*)calloc(8*sizeof(unsigned),sizeof(int));
    int *x = (int*)malloc(l*sizeof(int));
    int *y = (int*)malloc(l*sizeof(int));
    int *z = (int*)malloc(l*sizeof(int));
    int bitNum, t;
    double coord;
    dec2bin_int(num,rep,&bitNum);
    if(bitNum<3*l) {
        int dif = 3*l-bitNum;
        for(i=0;i<dif;i++) {
            ind[i] = 0;
        }
        for(i=dif;i<3*l;i++) {
            ind[i] = rep[i-dif];
        }
    } else {
        for(i=0;i<3*l;i++) {
            ind[i] = rep[i];
        }
    }
    bitDeintleave(ind,l,x,y,z);
    t = indArr2num(x,l,1);
    coord = pow(2,-l)*t+pow(2,-l-1);
    pnt.x = coord;
    t = indArr2num(y,l,1);
    coord = pow(2,-l)*t+pow(2,-l-1);
    pnt.y = coord;
    t = indArr2num(z,l,1);
    coord = pow(2,-l)*t+pow(2,-l-1);
    pnt.z = coord;
    free(ind);
    free(rep);
    free(x);
    free(y);
    free(z);
    return pnt;
}

int neighbors(const int num, const int l, int *numNeighbors, int *nbr)
{
    //neighbors of a box, not including itself
    int i, j, k, m, n=0;
    int dif;
    int *ind = (int*)malloc(3*l*sizeof(int));
    int *rep = (int*)calloc(3*l,sizeof(int)); //binary form of num
    int *x = (int*)malloc(l*sizeof(int)); //binary array for the x direction
    int *y = (int*)malloc(l*sizeof(int)); //binary array for the y direction
    int *z = (int*)malloc(l*sizeof(int)); //binary array for the z direction
    int *s = (int*)malloc(l*sizeof(int)); 
    int nbr_x[3], nbr_y[3], nbr_z[3]; //index of neighboring boxes in each dimension
    int numNbr_x, numNbr_y, numNbr_z;
    int bitNum, t;
    dec2bin_int(num,rep,&bitNum); //num to binary form
    if(bitNum<3*l) {
        dif = 3*l-bitNum;
        for(i=0;i<dif;i++) {
            ind[i] = 0;
        }
        for(i=dif;i<3*l;i++) {
            ind[i] = rep[i-dif];
        }
    } else {
        for(i=0;i<3*l;i++) {
            ind[i] = rep[i];
        }
    }
    //printf("The number of bits: %d\n",bitNum);
    //printIntArray(ind,3*l);
    
    bitDeintleave(ind,l,x,y,z);
    t = indArr2num(x,l,1);
    if(t==0) {
        numNbr_x = 2;
        nbr_x[0] = t;
        nbr_x[1] = t+1;
    } else {
        if(t==(int)(pow(2,l)-1)) {
            numNbr_x = 2;
            nbr_x[0] = t-1;
            nbr_x[1] = t;
        } else {
            numNbr_x = 3;
            nbr_x[0] = t-1;
            nbr_x[1] = t;
            nbr_x[2] = t+1;
        }
    }
    
    t = indArr2num(y,l,1);
    if(t == 0) {
        numNbr_y = 2;
        nbr_y[0] = t;
        nbr_y[1] = t+1;
    } else {
        if(t==(int)(pow(2,l)-1)) {
            numNbr_y = 2;
            nbr_y[0] = t-1;
            nbr_y[1] = t;
        } else {
            numNbr_y = 3;
            nbr_y[0] = t-1;
            nbr_y[1] = t;
            nbr_y[2] = t+1;
        }
    } 
    
    t = indArr2num(z,l,1);
    if(t == 0) {
        numNbr_z = 2;
        nbr_z[0] = t;
        nbr_z[1] = t+1;
    } else {
        if(t==(int)(pow(2,l)-1)) {
            numNbr_z = 2;
            nbr_z[0] = t-1;
            nbr_z[1] = t;
        } else {
            numNbr_z = 3;
            nbr_z[0] = t-1;
            nbr_z[1] = t;
            nbr_z[2] = t+1;
        }
    }
    //printf("The dimensional neighbor for x, y, and z: \n");
    //printIntArray(nbr_x,numNbr_x);
    //printIntArray(nbr_y,numNbr_y);
    //printIntArray(nbr_z,numNbr_z);
    *numNeighbors = numNbr_x*numNbr_y*numNbr_z-1;
    for(i=0;i<numNbr_x;i++) {
        dec2bin_int(nbr_x[i],rep,&bitNum);
        if(bitNum<l) {
            dif = l-bitNum;
            for(m=0;m<dif;m++) {
                x[m] = 0;
            }
            for(m=dif;m<l;m++) {
                x[m] = rep[m-dif];
            }
        }
        if(bitNum==l) {
            for(m=0;m<l;m++) {
                x[m] = rep[m];
            }
        }
        for(j=0;j<numNbr_y;j++) {
            dec2bin_int(nbr_y[j],rep,&bitNum);
            if(bitNum<l) {
                dif = l-bitNum;
                for(m=0;m<dif;m++) {
                    y[m] = 0;
                }
                for(m=dif;m<l;m++) {
                    y[m] = rep[m-dif];
                }
            }
            if(bitNum==l) {
                for(m=0;m<l;m++) {
                    y[m] = rep[m];
                }
            }
            for(k=0;k<numNbr_z;k++) {
                dec2bin_int(nbr_z[k],rep,&bitNum);
                if(bitNum<l) {
                    dif = l-bitNum;
                    for(m=0;m<dif;m++) {
                        z[m] = 0;
                    }
                    for(m=dif;m<l;m++) {
                        z[m] = rep[m-dif];
                    }
                }
                if(bitNum==l) {
                    for(m=0;m<l;m++) {
                        z[m] = rep[m];
                    }
                }
                bitIntleave(x,y,z,l,s);
                t = indArr2num(s,l,3);
                if(t!=num) {
                    nbr[n++] = t;
                }
            }
        }
    }
    //printf("n=%d\n",n);
    free(ind);
    free(rep);
    free(x);
    free(y);
    free(z);
    free(s);
    return EXIT_SUCCESS;
}

void createSet(const int *elems, const int numElems, int *set)
{
    int i, j;
    int flag = 0;
    set[0] = 0;
    for(i=0;i<numElems;i++) {
        for(j=0;j<set[0];j++) {
            if(elems[i]==set[j+1]) {
                flag = 1; //repeated elements
                break;
            }
        }
        if(flag==0) {
            set[++set[0]] = elems[i];
        }
        flag = 0;
    }
}

int isMember(const int t, const int *set)
{
    int i, flag = 0;
    for(i=1;i<=set[0];i++) {
        if(t==set[i]) {
            flag = 1;
            break;
        }
    }
    return flag;
}

int isEmpty(const int *set)
{
    if(set[0]==0) {
        return 1;
    } else {
        return 0;
    }
}

void intersection(const int *set1, const int *set2, int *set3)
{
    int i;
    set3[0] = 0;
    int n = set1[0]; //number of elements in set 1;
    for(i=1;i<=n;i++) {
        if(isMember(set1[i],set2)) {
            set3[++set3[0]] = set1[i];
        }
    }
}

void Union(const int *set1, const int *set2, int *set3) {
    int i;
    int n = set2[0];
    for(i=0;i<=set1[0];i++) {
        set3[i] = set1[i];
    }
    for(i=1;i<=n;i++) {
        if(!isMember(set2[i],set1)) {
            set3[++set3[0]] = set2[i];
        }
    }
}

void difference(const int *set1, const int *set2, int *set3) {
    //get elements in set 1 but not in set 2
    int i;
    set3[0] = 0;
    for(i=1;i<=set1[0];i++) {
        if(!isMember(set1[i],set2)) {
            set3[++set3[0]] = set1[i];
        }
    }
}

void pnts2numSet(const cartCoord_d *pnts, const int numPnts, const int l, int *set) {
    int i;
    int num[MAX];
    for(i=0;i<numPnts;i++) {
        num[i] = pnt2boxnum(pnts[i],l);
    }
    //printf("print array in pnts2numSet: \n");
    //printIntArray(num,numPnts);
    createSet(num,numPnts,set);
}



void sampleSpace(const int l, int *set) {
    //all boxes at level l
    int t = pow(8,l);
    set[0] = t;
    for(t=1;t<=set[0];t++) {
        set[t] = t-1;
    }
}

void I1(const int num, int *set) {
    set[0] = 1;
    set[1] = num;
}

void I2(const int num, const int l, int *set) {
    //the set of neighbors
    int nbr[27];
    int t;
    neighbors(num,l,&t,nbr); //not including (num,l)
    nbr[t] = num; //inclue the box itself
    createSet(nbr,t+1,set);
}

void I3(const int num, const int l, int *set) {
    int nbrSet[28];
    int *set_whole = (int*)malloc((pow(8,l)+1)*sizeof(int));
    sampleSpace(l,set_whole);
    I2(num,l,nbrSet); //neighborhood of (num,l)
    difference(set_whole,nbrSet,set);
    free(set_whole);
}

void I4(const int num, const int l, int *set) {
    int t, t_p, i, j;
    int num_p = parent(num); //number of the parent box
    int nbrs[27], nbrs_p[27]; 
    neighbors(num,l,&t,nbrs);
    nbrs[t] = num; //include the box itself
    neighbors(num_p,l-1,&t_p,nbrs_p); //neighbors of the parent box
    nbrs_p[t_p] = num_p; //include the box itsefl
    int *chldrn = (int*)malloc(8*(t_p+1)*sizeof(int)); //allocate memory for children
    //all children of the parent neighborhood
    for(i=0;i<t_p+1;i++) {
        for(j=0;j<8;j++)
            chldrn[8*i+j] = child(nbrs_p[i],j);
    }
    int *set_p = (int*)malloc((8*(t_p+1)+1)*sizeof(int));
    int set_c[28]; //current level
    createSet(chldrn,8*(t_p+1),set_p);
    createSet(nbrs,t+1,set_c);
    difference(set_p,set_c,set);
    free(chldrn);
    free(set_p);
}


void orderArray(const int *a, const int num, int *ind) {
    int i, j;
    int temp;
    for(i=0;i<num;i++) {
        ind[i] = i;
    }
    for(i=0;i<num;i++) {
        for(j=i+1;j<num;j++) {
            if(a[ind[i]]>a[ind[j]]) {
                temp = ind[i];
                ind[i] = ind[j];
                ind[j] = temp;
            }
        }
    }
}

void findNum(const char * filename, int *pV, int *pE) {
    /*Find the number of vertices and elements in the current geometry*/
    int i = 0, j = 0; // For saving number of vertices and elements
    char line[50]; // For reading each line of file
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Failed to open file.\n");
        exit(EXIT_FAILURE);
    }
    while (fgets(line, 49, fp) != NULL) {
        if (line[0] == 'v') {
            i++;
        }
        if (line[0] == 'f') {
            j++;
        }
    }
    *pV = i;
    *pE = j;
    fclose(fp);
}

int readOBJ(const char *filename, cartCoord_d *p, triElem *e) {
    int temp[3];
    FILE *fp = fopen(filename,"r");
    if (fp==NULL) {
        printf("Failed to open file.\n");
        exit(EXIT_FAILURE);
    }
    int i = 0, j = 0;
    char line[50];
    char type[5];
    while(fgets(line,49,fp)!=NULL) {
        //printf("%c\n",line[0]);
        if (line[0]=='v') {
            //printf("find 1 vertex.\n");
            sscanf(line,"%s %lf %lf %lf",type,&(p[i].x),
                    &(p[i].y),&(p[i].z));
            i++;
        }

        if (line[0]=='f') {
            sscanf(line,"%s %d %d %d",type,&temp[0],&temp[1],&temp[2]);
            e[j].node[0] = temp[0]-1;
            e[j].node[1] = temp[1]-1;
            e[j].node[2] = temp[2]-1;
            e[j].alpha = make_cuFloatComplex(0,0); // ca=0
            e[j].beta = make_cuFloatComplex(0,0); // cb=1
            e[j].gamma = make_cuFloatComplex(0,0); // cc=0
            j++;
        }
    }
    fclose(fp);
    return EXIT_SUCCESS;
}

void printPnts(const cartCoord_d *p, const int numPnts) {
    int i;
    for(i=0;i<numPnts;i++) {
        printf("(%f,%f,%f)\n",p[i].x,p[i].y,p[i].z);
    }
}

void printElems(const triElem *e, const int numElems) {
    int i;
    for(i=0;i<numElems;i++) {
        printf("(%d, %d, %d)\n",e[i].node[0],e[i].node[1],e[i].node[2]);
    }
}

int deterLmax(const cartCoord_d *pnts, const int numPnts, const int s)
{
    int i, j, l, t, l_max = 0;
    int a, b;
    const int l_avl = 10;
    int *pntInd = (int*)malloc(numPnts*sizeof(int));
    int *ind = (int*)malloc(numPnts*sizeof(int));
    for(i=0;i<numPnts;i++) {
        t = pnt2boxnum(pnts[i],l_avl);
        pntInd[i] = t;
    }
    /*
    for(i=0;i<numPnts;i++) {
        printf("%d ",pntInd[i]);
    }
     */
    orderArray(pntInd,numPnts,ind);
    //printf("print array in deterLmax: \n");
    //printIntArray(ind,numPnts);
    i = 0;
    j = s;
    while(j<numPnts) {
        l = l_avl;
        a = pntInd[ind[i]];
        b = pntInd[ind[j]];
        if(a==b) {
            printf("Integer type cannot accomodate the current accuracy.\n");
            return 11;
        }
        //at which level the two cartCoord_ds are in the same box
        while(a!=b) {
            l--;
            a = parent(a);
            b = parent(b);
        }
        l_max = max(l_max,l+1);
        i++;
        j++;
    }
    free(ind);
    free(pntInd);
    return l_max;
}

void findBoundingCube(const cartCoord_d* pnts, const int numPnts, cartCoord_d* pnts_b, 
        double* d)
{
    int i;
    double x_min, x_max, y_min, y_max, z_min, z_max;
    double d_x, d_y, d_z, d_max;
    x_min = pnts[0].x;
    x_max = pnts[0].x;
    y_min = pnts[0].y;
    y_max = pnts[0].y;
    z_min = pnts[0].y;
    z_max = pnts[0].y;
    for(i=1;i<numPnts;i++) {
        if(pnts[i].x < x_min) {
            x_min = pnts[i].x;
        }
        if(pnts[i].x > x_max) {
            x_max = pnts[i].x;
        }
        if(pnts[i].y < y_min) {
            y_min = pnts[i].y;
        }
        if(pnts[i].y > y_max) {
            y_max = pnts[i].y;
        }
        if(pnts[i].z < z_min) {
            z_min = pnts[i].z;
        }
        if(pnts[i].z > z_max) {
            z_max = pnts[i].z;
        }
    }
    x_max += 0.01;
    x_min -= 0.01;
    y_max += 0.01;
    y_min -= 0.01;
    z_max += 0.01;
    z_min -= 0.01;
    d_x = x_max-x_min;
    d_y = y_max-y_min;
    d_z = z_max-z_min;
    d_max = max(max(d_x,d_y),d_z);
    x_max += (d_max-d_x)/2;
    x_min -= (d_max-d_x)/2;
    y_max += (d_max-d_y)/2;
    y_min -= (d_max-d_y)/2;
    z_max += (d_max-d_z)/2;
    z_min -= (d_max-d_z)/2;
    pnts_b[0].x = x_min;
    pnts_b[0].y = y_min;
    pnts_b[0].z = z_min;
    pnts_b[1].x = x_min;
    pnts_b[1].y = y_min;
    pnts_b[1].z = z_max;
    pnts_b[2].x = x_min;
    pnts_b[2].y = y_max;
    pnts_b[2].z = z_min;
    pnts_b[3].x = x_min;
    pnts_b[3].y = y_max;
    pnts_b[3].z = z_max;
    pnts_b[4].x = x_max;
    pnts_b[4].y = y_min;
    pnts_b[4].z = z_min;
    pnts_b[5].x = x_max;
    pnts_b[5].y = y_min;
    pnts_b[5].z = z_max;
    pnts_b[6].x = x_max;
    pnts_b[6].y = y_max;
    pnts_b[6].z = z_min;
    pnts_b[7].x = x_max;
    pnts_b[7].y = y_max;
    pnts_b[7].z = z_max;
    *d = d_max;
}

/*
void srcBoxes(const cartCoord_d *pnts, const triElem *elems, const int numElems, 
        const int s, int *srcBoxSet, int *lmax, double *D, cartCoord_d *pnt_min) {
    int i;
    vector v1, v2, v3, v;
    cartCoord_d *pnts_ctr = (cartCoord_d*)malloc(numElems*sizeof(cartCoord_d));
    cartCoord_d *pnts_bnd = (cartCoord_d*)malloc(8*sizeof(cartCoord_d));
    cartCoord_d *pnts_scaled = (cartCoord_d*)malloc(numElems*sizeof(cartCoord_d));
    for(i=0;i<numElems;i++) {
        v1 = pnt2vec(pnts[elems[i].nodes[0]]);
        v2 = pnt2vec(pnts[elems[i].nodes[1]]);
        v3 = pnt2vec(pnts[elems[i].nodes[2]]);
        v = triCentroid(v1,v2,v3);
        pnts_ctr[i] = vec2pnt(v);
    }
    findBoundingCube(pnts_ctr,numElems,pnts_bnd,D);
    *pnt_min = pnts_bnd[0];
    scalePnts(pnts_ctr,numElems,&pnts_bnd[0],*D,pnts_scaled);
    *lmax = deterLmax(pnts_scaled,numElems,s);
    pnts2numSet(pnts_scaled,numElems,*lmax,srcBoxSet);
}

int truncNum(const double k, const double eps, const double sigma, 
        const double a) {
    double p_lo, p_hi, p, temp_d[2];
    temp_d[0] = eps*pow(1-1.0/sigma,1.5);
    temp_d[1] = log(temp_d[0])/log(sigma);
    p_lo = 1-temp_d[1];
    
    temp_d[0] = k*a;
    temp_d[1] = pow(3*log(1.0/eps),1.5)/2;
    p_hi = temp_d[0]+temp_d[1]*pow(temp_d[0],1.0/3);
    
    temp_d[0] = pow(p_lo,4)+pow(p_hi,4);
    p = ceil(pow(temp_d[0],0.25));
    
    //if(p<10) {
    //    p = 10;
    //}
    
    return (int)p;
}

int truncNum_2(const double wavNum, const double eps, const double sigma, 
        const double a) {
    double p;
    double ka_s = 3.0/(pow(2,1.5)*pow(sigma-1,1.5))*log(1.0/(eps*sigma));
    double p_s = 3.0*sigma/(pow(2,1.5)*pow(sigma-1,1.5))*log(1.0/(eps*sigma));
    if(wavNum*a<=ka_s) {
        p = p_s;
    } else {
        p = wavNum*a+0.5*pow(3*log(1.0/(eps*sigma)),2.0/3)*pow(wavNum*a,1.0/3);
    }
    return (int)ceil(p);
}

double descale_1d(const double a, const double D, const double v_min) {
    return D*a+v_min;
}

void prtlvlSet(const int *X, const int l, int *X_n) {
    if(l == 0) {
        return;
    }
    int i, num = 0, X_t[MAX];
    for(i=0;i<X[0];i++) {
        if(X[i+1]>=pow(8,l)) {
            printf("Error in box number.\n");
            return;
        }
        X_t[i] = parent(X[i+1]);
        num++;
    }
    createSet(X_t,num,X_n);
}

int findSetInd(const int *X, const int num) {
    if(X[0]==0) {
        printf("Empty set.\n");
        return -1;
    }
    int i, t = -1;
    for(i=1;i<=X[0];i++) {
        if(X[i]==num) {
            t = i-1;
            break;
        }
    }
    if(t==-1) {
        printf("Not in the set.\n");
    }
    return t;
}

void cpySet(const int *X, int *Y) {
    int i;
    for(i=0;i<=X[0];i++) {
        Y[i] = X[i];
    }
}

void FMMLvlSet_s(const int *X, const int lmax, int ***pSet) {
    int l;
    const int lmin = 2;
    *pSet = (int**)malloc((lmax-2+1)*sizeof(int*));
    int set_temp[MAX];
    (*pSet)[0] = (int*)malloc((X[0]+1)*sizeof(int));
    cpySet(X,(*pSet)[0]);
    for(l=lmax-1;l>=lmin;l--) {
        prtlvlSet((*pSet)[lmax-l-1],l+1,set_temp);
        (*pSet)[lmax-l] = (int*)malloc((set_temp[0]+1)*sizeof(int));
        cpySet(set_temp,(*pSet)[lmax-l]);
    }
}

void FMMLvlSet_e(const int *Y, const int lmax, int ***pSet) {
    int l;
    const int lmin = 2;
    int set_temp[MAX];
    *pSet = (int**)malloc((lmax-2+1)*sizeof(int*));
    //printf("allocated memory.\n");
    (*pSet)[lmax-2] = (int*)malloc((Y[0]+1)*sizeof(int));
    cpySet(Y,(*pSet)[lmax-2]);
    for(l=lmax-1;l>=lmin;l--) {
        prtlvlSet((*pSet)[l+1-lmin],l+1,set_temp);
        (*pSet)[l-lmin] = (int*)malloc((set_temp[0]+1)*sizeof(int));
        cpySet(set_temp,(*pSet)[l-lmin]);
    }
    //printf("completed FMMLvlSet_e\n");
}

void sortSet(int *set) {
    int i, j, temp;
    for(i=1;i<=set[0];i++) {
        for(j=i;j<=set[0];j++) {
            if(set[i]>set[j]) {
                temp = set[j];
                set[j] = set[i];
                set[i] = temp;
            }
        }
    }
}
*/



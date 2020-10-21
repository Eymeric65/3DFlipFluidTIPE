#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CLASSsim/ParticleClass.h"
#include "CLASSsim/FLIPimpl.h"
#include <helper_cuda.h> 

__device__ unsigned int getGridInd(unsigned int indiceX, unsigned int indiceY, unsigned int indiceZ, uint3 BoxIndice)
{
    return indiceZ + indiceY * BoxIndice.z + indiceX * BoxIndice.z * BoxIndice.y;
}

__device__ float3 getDiv(unsigned int x, unsigned int y, unsigned int z,float3* Gridspeed, uint3 BoxIndice,float tsize)
{

    if (x > 0 && y > 0 && z > 0)
    {
        unsigned int p = getGridInd(x, y, z, BoxIndice);
        unsigned int xm = getGridInd(x - 1, y, z, BoxIndice);

        unsigned int ym = getGridInd(x, y - 1, z, BoxIndice);

        unsigned int zm = getGridInd(x, y, z - 1, BoxIndice);

        float xr = Gridspeed[p].x - Gridspeed[xm].x;

        float yr = Gridspeed[p].y - Gridspeed[ym].y;

        float zr = Gridspeed[p].z - Gridspeed[zm].z;

        return make_float3(xr / tsize, yr / tsize, zr / tsize);
    }
    else
    {
        return make_float3(0.69, 0.69, 0.69);
    }


}

__device__ int staggeredGrid(float gc, float pc)
{
    if (pc - gc >= 0)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}



__global__ void gridnormal(uint3 boxind, float3* gridspeed, float* gridcounter,unsigned int* type)
{
    unsigned int index = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z,boxind);
    //printf("la cases %d %d %d \n", blockIdx.x, blockIdx.y, blockIdx.z);
    //printf("case %d \n", index);

    if (blockIdx.x == 0 || blockIdx.x == boxind.x-1 ||
        blockIdx.y == 0 || blockIdx.y == boxind.y-1 ||
        blockIdx.z == 0 || blockIdx.z == boxind.z-1)
    {
        type[index] = 0;
    }
    else if (gridcounter[index] != 0)
    {
        gridspeed[index] = make_float3(gridspeed[index].x/ gridcounter[index], gridspeed[index].y/ gridcounter[index], gridspeed[index].z/ gridcounter[index]);
        //printf("la case %d possède des coordonnées x: %f , y: %f , z: %f \n",index, gridspeed[index].x, gridspeed[index].y, gridspeed[index].z);

        type[index] = 1;
    }
    else
    {
        type[index] = 2;
    }

    ///float3 vel = getDiv(blockIdx.x, blockIdx.y, blockIdx.z, gridspeed, boxind, tsize);

    //printf("case %d , %d, %d possede type %d \n", blockIdx.x, blockIdx.y, blockIdx.z,type[index]);
}

__global__ void addforces_k(uint3 boxind, float3* gridspeed, unsigned int* type,float tstep)
{
    unsigned int index = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z, boxind);

    if (type[index] == 1)
    {
        gridspeed[index].y += -9.81 * tstep; 


    }

}

__global__ void boundariescondition(uint3 boxind, float3* gridspeed)
{
    unsigned int index = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z, boxind);

    if (blockIdx.x == 0 || blockIdx.x == (boxind.x - 2))
    {
        gridspeed[index].x = 0;
    }

    if (blockIdx.y == 0 || blockIdx.y == (boxind.y - 2))
    {
        gridspeed[index].y = 0;
    }

    if (blockIdx.z == 0 || blockIdx.z == (boxind.z - 2))
    {
        gridspeed[index].z = 0;
    }
}

__global__ void addpressure_k(uint3 boxind, float3* gridspeed, float* gridpressure, unsigned int* type, float density,float tsize)
{
   unsigned int index = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z, boxind);

   if (type[index] == 1)
   {

       unsigned int indexXm = index - boxind.z * boxind.y;
       unsigned int indexXp = index + boxind.z * boxind.y;


       unsigned int indexYm = index - boxind.z;
       unsigned int indexYp = index + boxind.z;


       unsigned int indexZm = index - 1;
       unsigned int indexZp = index + 1;

       gridspeed[index].x -= (gridpressure[indexXp]- gridpressure[index])/ tsize;
       gridspeed[index].y -= (gridpressure[indexYp] - gridpressure[index]) / tsize;
       gridspeed[index].z -= (gridpressure[indexZp] - gridpressure[index]) /tsize;

       float divg = (gridspeed[index].x - gridspeed[indexXm].x + gridspeed[index].y - gridspeed[indexYm].y + gridspeed[index].z - gridspeed[indexZm].z) / tsize;
       if (abs(divg)>1) { printf("la divergence est de : %f \n", divg); }
   }

}

__global__ void JacobiIterationForPressure(uint3 boxind, float3* gridspeed,float* gridpressureB,float* gridpressureA, unsigned int* type,float tsize,float density,float* GridCounter )
{
    unsigned int index = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z, boxind);
    if (type[index] == 1)
    {
        //float3 divg = getDiv(blockIdx.x, blockIdx.y, blockIdx.z, gridspeed, boxind, tsize);
        unsigned int boundCounter = 0;


        unsigned int indexXm = index - boxind.z * boxind.y;
        unsigned int indexXp = index + boxind.z * boxind.y;


        unsigned int indexYm = index - boxind.z;
        unsigned int indexYp = index + boxind.z;


        unsigned int indexZm = index - 1;
        unsigned int indexZp = index + 1;

        float divg = ( gridspeed[index].x - gridspeed[indexXm].x + gridspeed[index].y - gridspeed[indexYm].y + gridspeed[index].z - gridspeed[indexZm].z)/tsize  ;

        float pxm = 0;
        if (blockIdx.x - 1 != 0)
        {
            boundCounter++;
            float pxm = gridpressureB[indexXm];
        }

        float pxp = 0;
        if (blockIdx.x + 1 != boxind.x-1)
        {
            boundCounter++;
            float pxp = gridpressureB[indexXp];
        }

        float pym = 0;
        if (blockIdx.y - 1 != 0)
        {
            boundCounter++;
            float pym = gridpressureB[indexYm];
        }

        float pyp = 0;
        if (blockIdx.y + 1 != boxind.y - 1)
        {
            boundCounter++;
            float pyp = gridpressureB[indexYp];
        }

        float pzm = 0;
        if (blockIdx.z - 1 != 0)
        {
            boundCounter++;
            float pzm = gridpressureB[indexZm];
        }

        float pzp = 0;
        if (blockIdx.z + 1 != boxind.z - 1)
        {
            boundCounter++;
            float pzp = gridpressureB[indexZp];
        }
        



        gridpressureA[index] = (pxm+pxp+pym+pyp+pzm+pzp - divg)/ boundCounter;
        


        if (index == 1003)
        {
            //printf("il y a %f \n", fmaxf(GridCounter[index] - density, 0));
           //printf("la div est %f %f %f \n", divgx, divgy, divgz);
        }

    }
}

__global__ void PressureCopy(uint3 boxind,float* gridpressureB, float* gridpressureA)
{
    unsigned int index = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z, boxind);
    gridpressureB[index] = gridpressureA[index];
}

__global__ void TransferToGrid(unsigned int partcount, float3* pos, float3* vit, float3* gridspeed, float* gridcounter, float tsize, uint3 BoxIndice)
{

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < partcount; i += stride)
    {

        //printf("index is %d \n", index);

        int gridindexX = (int)(pos[index].x / tsize);
        int gridindexY = (int)(pos[index].y / tsize);
        int gridindexZ = (int)(pos[index].z / tsize);


        
        if (pos[index].x < 0 || pos[index].x > (BoxIndice.x - 1)* tsize ||
            pos[index].y < 0 || pos[index].y > (BoxIndice.y - 1)* tsize ||
            pos[index].z < 0 || pos[index].z > (BoxIndice.z - 1)* tsize)
        {
            //printf("particule number %d out of bound with indice : %d %d %d\n", index, gridindexX, gridindexY, gridindexZ);
        }
        else
        {
            ///vit[index] = make_float3(1, 1, 1); test purpose only
            //printf("la case %d \n", gridind);

            float centerposX = gridindexX * tsize;
            float centerposY = gridindexY * tsize;
            float centerposZ = gridindexZ * tsize;

            unsigned int gridindX = getGridInd(gridindexX + staggeredGrid(centerposX, pos[index].x), gridindexY, gridindexZ, BoxIndice);
            unsigned int gridindY = getGridInd(gridindexX, gridindexY + staggeredGrid(centerposY, pos[index].y), gridindexZ, BoxIndice);
            unsigned int gridindZ = getGridInd(gridindexX, gridindexY, gridindexZ + staggeredGrid(centerposZ, pos[index].z), BoxIndice);

            unsigned int gridind = getGridInd(gridindexX, gridindexY, gridindexZ, BoxIndice);

            atomicAdd(&(gridspeed[gridindX].x), vit[index].x);
            atomicAdd(&gridspeed[gridindY].y, vit[index].y);
            atomicAdd(&gridspeed[gridindZ].z, vit[index].z);



            atomicAdd(&gridcounter[gridind], 1.0f);

            /*
            if (index == 1999)
            {
                printf("atomic test %f , %f \n", vit[index].y, gridcounter[gridind]);
            }*/

        }
        //printf("indice de la particule dans la grille %d %d \n ",index, getGridInd(gridindexX, gridindexY, gridindexZ, BoxIndice));

        //pos[index] = make_float3(3, 3, 3);s

        //printf("la particule %d possède des coordonnées x: %f , y: %f , z: %f \n",index, pos[index].x, pos[index].y, pos[index].z);

    }


}

// ce transfert de vitesse est vraiment plutot brutal méthode PIC
__global__ void TransfertToParticule(unsigned int partcount, float3* pos, float3* vit, float3* gridspeed, float* gridcounter, float tsize, uint3 BoxIndice)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < partcount; i += stride)
    {

        //printf("index is %d \n", index);

        int gridindexX = (int)(pos[index].x / tsize);
        int gridindexY = (int)(pos[index].y / tsize);
        int gridindexZ = (int)(pos[index].z / tsize);

        if (pos[index].x < 0 || pos[index].x >(BoxIndice.x - 1)* tsize ||
            pos[index].y < 0 || pos[index].y >(BoxIndice.y - 1) * tsize ||
            pos[index].z < 0 || pos[index].z >(BoxIndice.z - 1) * tsize)
        {
            //printf("particule number %d out of bound with indice : %d %d %d\n", index, gridindexX, gridindexY, gridindexZ);
        }
        else
        {



            //printf("la case %d \n", gridind);

            float centerposX = gridindexX * tsize;
            float centerposY = gridindexY * tsize;
            float centerposZ = gridindexZ * tsize;

            unsigned int gridindX = getGridInd(gridindexX + staggeredGrid(centerposX, pos[index].x), gridindexY, gridindexZ, BoxIndice);
            unsigned int gridindY = getGridInd(gridindexX, gridindexY + staggeredGrid(centerposY, pos[index].y), gridindexZ, BoxIndice);
            unsigned int gridindZ = getGridInd(gridindexX, gridindexY, gridindexZ + staggeredGrid(centerposZ, pos[index].z), BoxIndice);

            vit[index] = make_float3(gridspeed[gridindX].x, gridspeed[gridindY].y, gridspeed[gridindZ].z); // direct assigmement

            //printf("la vitesse de la particule %d est %f %f %f \n", index, vit[index].x, vit[index].y, vit[index].z);

        }

        //printf("indice de la particule dans la grille %d %d \n ",index, getGridInd(gridindexX, gridindexY, gridindexZ, BoxIndice));

        //pos[index] = make_float3(3, 3, 3);

        //printf("la particule %d possède des coordonnées x: %f , y: %f , z: %f \n",index, pos[index].x, pos[index].y, pos[index].z);

    }

}
// -------------------------------------------------------------------------------------------
extern "C"
void TrToGr(ParticleSystem *partEngine, FlipSim * flipEngine)
{
    //std::cout <<"nombre de particule "<< partEngine->PartCount << std::endl;

    TransferToGrid << <1000, 1024 >> > (partEngine->PartCount,
                                        partEngine->Partpos,
                                        partEngine->Partvit,
                                        flipEngine->GridSpeed,
                                        flipEngine->GridCounter,
                                        flipEngine->tileSize,
                                        flipEngine->BoxIndice);

    uint3 box = flipEngine->BoxIndice;
    
    dim3 numblocks(box.x, box.y, box.z);
    
    gridnormal << <numblocks, 1 >> > (  box,
                                        flipEngine->GridSpeed,
                                        flipEngine->GridCounter,
                                        flipEngine->type);
}

extern "C"
void addforces(ParticleSystem * partEngine, FlipSim * flipEngine)
{
    uint3 box = flipEngine->BoxIndice;

    dim3 numblocks(box.x, box.y, box.z);

    addforces_k << <numblocks, 1 >> > (box,
                                      flipEngine->GridSpeed,
                                      flipEngine->type,
                                      flipEngine->timestep);

    boundariescondition << <numblocks, 1 >> > (box, flipEngine->GridSpeed);

}

extern "C"
void TrToPr(ParticleSystem * partEngine, FlipSim * flipEngine)
{
    TransfertToParticule << <1000, 1024 >> > (partEngine->PartCount,
        partEngine->Partpos,
        partEngine->Partvit,
        flipEngine->GridSpeed,
        flipEngine->GridCounter,
        flipEngine->tileSize,
        flipEngine->BoxIndice);

}

extern "C"
void JacobiIter( FlipSim * flipEngine, unsigned int stepNb)
{

    //cudaMemset(flipEngine->GridPressureB, 0, flipEngine->IndiceCount * sizeof(float3));

    uint3 box = flipEngine->BoxIndice;

    dim3 numblocks(box.x, box.y, box.z);

    for (int i = 0; i < stepNb; i += 1)
    {
        
        JacobiIterationForPressure << <numblocks, 1 >> > (box,
            flipEngine->GridSpeed,
            flipEngine->GridPressureB,
            flipEngine->GridPressureA,
            flipEngine->type,
            flipEngine->tileSize,
            10,
            flipEngine->GridCounter);
            
        PressureCopy<<<numblocks,1>>>(box, flipEngine->GridPressureB, flipEngine->GridPressureA);

    }

}

extern "C"
void AddPressureForce(FlipSim * flipEngine)
{
    uint3 box = flipEngine->BoxIndice;

    dim3 numblocks(box.x, box.y, box.z);

    addpressure_k<<<numblocks ,1>>>(box, flipEngine->GridSpeed, flipEngine->GridPressureA, flipEngine->type, 1, flipEngine->tileSize);


}

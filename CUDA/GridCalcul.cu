#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CLASSsim/ParticleClass.h"
#include "CLASSsim/FLIPimpl.h"
#include <helper_cuda.h> 

__device__ unsigned int getGridInd(unsigned int indiceX, unsigned int indiceY,unsigned int indiceZ,uint3 BoxIndice)
{
    return indiceZ + indiceY*BoxIndice.z + indiceX*BoxIndice.z*BoxIndice.y;
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

__global__ void TransferToGrid(unsigned int partcount,float3* pos,float3* vit, float3* gridspeed,float* gridcounter,float tsize,uint3 BoxIndice)
{
	
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < partcount; i += stride)
    {

        //printf("index is %d \n", index);
        
        int gridindexX = (int)(pos[index].x / tsize);
        int gridindexY = (int)(pos[index].y / tsize);
        int gridindexZ = (int)(pos[index].z / tsize);

        

        if (gridindexX<1 || gridindexX>=(BoxIndice.x-1) ||
            gridindexY<1 || gridindexY>=(BoxIndice.y-1) ||
            gridindexZ<1 || gridindexZ>=(BoxIndice.z-1))
        {
            printf("particule number %d out of bound with indice : %d %d %d\n", index, gridindexX, gridindexY, gridindexZ);
        }
        else
        {
            vit[index] = make_float3(1, 1, 1);
            //printf("la case %d \n", gridind);

            float centerposX = gridindexX * tsize;
            float centerposY = gridindexY * tsize;
            float centerposZ = gridindexZ * tsize;

            unsigned int gridindX = getGridInd(gridindexX + staggeredGrid(centerposX, pos[index].x), gridindexY, gridindexZ, BoxIndice);
            unsigned int gridindY = getGridInd(gridindexX , gridindexY + staggeredGrid(centerposY, pos[index].y), gridindexZ, BoxIndice);
            unsigned int gridindZ = getGridInd(gridindexX , gridindexY, gridindexZ + staggeredGrid(centerposZ, pos[index].z), BoxIndice);

            unsigned int gridind = getGridInd(gridindexX , gridindexY, gridindexZ, BoxIndice);

            atomicAdd(&gridspeed[gridindX].x, vit[index].x);
            atomicAdd(&gridspeed[gridindY].y, vit[index].y);
            atomicAdd(&gridspeed[gridindZ].z, vit[index].z);

            atomicAdd(&gridcounter[gridind], 1.0f);
        }

        //printf("indice de la particule dans la grille %d %d \n ",index, getGridInd(gridindexX, gridindexY, gridindexZ, BoxIndice));

        //pos[index] = make_float3(3, 3, 3);s
         
        //printf("la particule %d possède des coordonnées x: %f , y: %f , z: %f \n",index, pos[index].x, pos[index].y, pos[index].z);

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
        gridspeed[index].x += -9.81 * tstep; 
    }

}

extern "C"
void TrToGr(ParticleSystem *partEngine, FlipSim * flipEngine)
{
    std::cout <<"nombre de particule "<< partEngine->PartCount << std::endl;

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

}

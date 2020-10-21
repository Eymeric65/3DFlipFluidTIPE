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

        

        if (gridindexX<0 || gridindexX>BoxIndice.x ||
            gridindexY<0 || gridindexY>BoxIndice.y ||
            gridindexZ<0 || gridindexZ>BoxIndice.z)
        {
            printf("particule number %d out of bound with indice : %d %d %d\n", index, gridindexX, gridindexY, gridindexZ);
        }
        else
        {
            unsigned int gridind = getGridInd(gridindexX, gridindexY, gridindexZ, BoxIndice);
            //printf("la case %d \n", gridind);

            atomicAdd(&gridspeed[gridind].x, vit[index].x);
            atomicAdd(&gridspeed[gridind].y, vit[index].y);
            atomicAdd(&gridspeed[gridind].z, vit[index].z);

            atomicAdd(&gridcounter[gridind], 1.0f);
        }

        //printf("indice de la particule dans la grille %d %d \n ",index, getGridInd(gridindexX, gridindexY, gridindexZ, BoxIndice));

        //pos[index] = make_float3(3, 3, 3);s
         
        //printf("la particule %d possède des coordonnées x: %f , y: %f , z: %f \n",index, pos[index].x, pos[index].y, pos[index].z);

    }
	

}

__global__ void gridnormal(uint3 boxind, float3* gridspeed, float* gridcounter)
{
    unsigned int index = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z,boxind);
    //printf("la cases %d %d %d \n", blockIdx.x, blockIdx.y, blockIdx.z);
    //printf("case %d \n", index);
    if (gridcounter[index] != 0)
    {
        gridspeed[index] = make_float3(gridspeed[index].x/ gridcounter[index], gridspeed[index].y/ gridcounter[index], gridspeed[index].z/ gridcounter[index]);
        //printf("la case %d possède des coordonnées x: %f , y: %f , z: %f \n",index, gridspeed[index].x, gridspeed[index].y, gridspeed[index].z);
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
                                                    flipEngine->GridCounter);
    
}

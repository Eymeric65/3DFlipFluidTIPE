#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CLASSsim/ParticleClass.h"
#include "CLASSsim/FLIPimpl.h"


__global__ void TransferToGrid(unsigned int partcount,float3* pos, float3* gridspeed,float tsize,uint3 BoxIndice)
{
	
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < partcount; i += stride)
    {
        
        int gridindexX = (int)(pos[index].x / tsize);
        int gridindexY = (int)(pos[index].y / tsize);
        int gridindexZ = (int)(pos[index].z / tsize);

        //printf("index is %d \n",index);



    }
	

}

extern "C"
void TrToGr(ParticleSystem *partEngine, FlipSim * flipEngine)
{
    unsigned int partcount = partEngine->PartCount; 

    //std::cout << flipEngine->tileSize << std::endl;

    TransferToGrid << <1000, 1024 >> > (partcount,
                                        partEngine->pos,
                                        flipEngine->GridSpeed,
                                        flipEngine->GridCounter,
                                        flipEngine->tileSize);

}

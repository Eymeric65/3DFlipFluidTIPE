#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CLASSsim/ParticleClass.h"
#include <helper_cuda.h> 

__global__ void EulerIntegration(unsigned int partcount,float3* pos, float3* vit, float tstep,float3 boxsize,float tsize)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < partcount; i += stride)
    {
        pos[index].x += vit[index].x * tstep;
        pos[index].y += vit[index].y * tstep;
        pos[index].z += vit[index].z * tstep;

        if (pos[index].x < tsize)
        {
            pos[index].x = tsize*1.1;
        }
        if (pos[index].x > boxsize.x-tsize)
        {
            pos[index].x = boxsize.x-tsize * 1.1;
        }

        if (pos[index].y < tsize)
        {
            pos[index].y = tsize * 1.1;
        }
        if (pos[index].y > boxsize.y- tsize)
        {
            pos[index].y = boxsize.y- tsize * 1.1;
        }

        if (pos[index].z < tsize)
        {
            pos[index].z = tsize * 1.1;
        }
        if (pos[index].z > boxsize.z- tsize)
        {
            pos[index].z = boxsize.z- tsize * 1.1;
        }
    }
}

extern "C"
void eulercompute(ParticleSystem * partEngine)
{
    EulerIntegration<<<1000,1024>>>(partEngine->PartCount,
                                    partEngine->Partpos,
                                    partEngine->Partvit,
                                    partEngine->TimeStep,
                                    partEngine->Boxsize,
                                    partEngine->tilesize);
}
#include "ParticleClass.h"
#include <iostream>
#include <helper_cuda.h> 

ParticleSystem::ParticleSystem(int partcount)
{
    PartCount = partcount;

    checkCudaErrors(cudaMalloc(&Partvit, PartCount * sizeof(float3)) );
    checkCudaErrors(cudaMemset(Partvit, 0, PartCount * sizeof(float3)) );

    //cudaMalloc(&pos, PartCount * sizeof(float3));


}

void ParticleSystem::StartCompute()
{
    checkCudaErrors( cudaGraphicsMapResources(1, &cuda_pos_resource, 0) );

    checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void**)&Partpos, &num_bytes_pos, cuda_pos_resource) );

}

void ParticleSystem::EndCompute()
{
    checkCudaErrors( cudaGraphicsUnmapResources(1, &cuda_pos_resource, 0) );

    //std::cout << "la taille est " << vit[0].x << std::endl;
    
}

void ParticleSystem::linkPos(GLuint buffer)
{
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pos_resource, buffer, cudaGraphicsRegisterFlagsNone) );
}

void ParticleSystem::endSystem()
{
    cudaFree(Partvit);

}

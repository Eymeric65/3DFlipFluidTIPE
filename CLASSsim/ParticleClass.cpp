#include "ParticleClass.h"

ParticleSystem::ParticleSystem(int partcount)
{
    PartCount = partcount;

    cudaMalloc(&vit, PartCount * sizeof(float3));
    cudaMemset(vit, 0, PartCount * sizeof(float3));

}

void ParticleSystem::Compute()
{
    cudaGraphicsMapResources(1, &cuda_pos_resource, 0);

    cudaGraphicsResourceGetMappedPointer((void**)&pos, &num_bytes_pos, cuda_pos_resource);

    cudaGraphicsUnmapResources(1, &cuda_pos_resource, 0);

}

void ParticleSystem::linkPos(GLuint buffer)
{
    cudaGraphicsGLRegisterBuffer(&cuda_pos_resource, buffer, cudaGraphicsRegisterFlagsNone);
}

void ParticleSystem::endSystem()
{
    cudaFree(vit);

}

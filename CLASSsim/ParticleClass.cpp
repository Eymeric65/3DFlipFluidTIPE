#include "ParticleClass.h"
#include <iostream>
#include <helper_cuda.h> 

extern "C" void eulercompute(ParticleSystem * partEngine);

ParticleSystem::ParticleSystem(int partcount,float tstep)
{
    PartCount = partcount;

    TimeStep = tstep;

    checkCudaErrors(cudaMalloc(&Partvit, PartCount * sizeof(float3)) );
    checkCudaErrors(cudaMemset(Partvit, 0, PartCount * sizeof(float3)) );
    std::cout << "reset partvit" << std::endl;
    //cudaMalloc(&pos, PartCount * sizeof(float3));


}

void ParticleSystem::StartCompute()
{
    checkCudaErrors( cudaGraphicsMapResources(1, &cuda_pos_resource, 0) );

    checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void**)&Partpos, &num_bytes_pos, cuda_pos_resource) );

}

void ParticleSystem::Compute()
{
    eulercompute(this);
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

#include "FLIPimpl.h"
#include <assert.h>
#include <iostream>


extern "C" void TrToGr( FlipSim * flipEngine);

extern "C" void addforces( FlipSim * flipEngine);

extern "C" void TrToPr( FlipSim * flipEngine);

extern "C" void JacobiIter(FlipSim * flipEngine,unsigned int stepNb);


extern "C" void AddPressureForce(FlipSim * flipEngine);

extern "C" void eulercompute(FlipSim * flipEngine);

FlipSim::FlipSim(float width, float height,float length, float tsize, unsigned int partcount,float tstep)
{

	PartCount = partcount;

	TimeStep = tstep;



	std::cout << "reset partvit" << std::endl;
	//cudaMalloc(&pos, PartCount * sizeof(float3));

	BoxSize = make_float3(width, height, length);

	tileSize = tsize;



	BoxIndice = make_uint3((int)(BoxSize.x / tileSize),
							(int)(BoxSize.y/ tileSize),
							(int)(BoxSize.z / tileSize)  );

	assert( "la taille de la case n'est pas un multiple de la taille de la boite"&&
		    ((BoxSize.x / tileSize) == BoxIndice.x) &&
			((BoxSize.y / tileSize) == BoxIndice.y) &&
			((BoxSize.z / tileSize) == BoxIndice.z)
			);


	IndiceCount = BoxIndice.x * BoxIndice.y * BoxIndice.z;

	printf("il y a %d cases \n",IndiceCount);

	printf("la boite possède (%d;%d;%d)cases \n", BoxIndice.x, BoxIndice.y, BoxIndice.z);

	cudaMalloc(&Partvit, PartCount * sizeof(float3));
	cudaMemset(Partvit, 0, PartCount * sizeof(float3));

	cudaMalloc(&GridSpeed, IndiceCount * sizeof(float3));
	cudaMemset(GridSpeed, 0, IndiceCount * sizeof(float3));

	cudaMalloc(&GridWeight, IndiceCount * sizeof(float));
	cudaMemset(GridWeight, 0, IndiceCount * sizeof(float));

	cudaMalloc(&GridPressureB, IndiceCount * sizeof(float3));
	cudaMemset(GridPressureB, 0, IndiceCount * sizeof(float3));

	cudaMalloc(&GridPressureA, IndiceCount * sizeof(float3));

	cudaMalloc(&type, IndiceCount * sizeof(unsigned int));



}

void FlipSim::TransferToGrid()
{
	cudaMemset(GridWeight, 0, IndiceCount * sizeof(float));
	cudaMemset(GridSpeed, 0, IndiceCount * sizeof(float3));
	TrToGr(this);
}

void FlipSim::TransferToParticule()
{
	TrToPr(this);
}

void FlipSim::AddExternalForces()
{
	addforces( this);
}

void FlipSim::PressureCompute()
{
	cudaMemset(GridPressureB, 0, IndiceCount * sizeof(float));
	JacobiIter(this, 50);
}

void FlipSim::AddPressure()
{
	AddPressureForce(this);
}

void FlipSim::endSim()
{
	cudaFree(GridSpeed);
	cudaFree(GridWeight);
	cudaFree(type);
	cudaFree(GridPressureB);
	cudaFree(GridPressureA);

	cudaFree(Partvit);
}

void FlipSim::StartCompute()
{
	cudaGraphicsMapResources(1, &cuda_pos_resource, 0);

	cudaGraphicsResourceGetMappedPointer((void**)&Partpos, &num_bytes_pos, cuda_pos_resource);

}

void FlipSim::linkPos(GLuint buffer)
{
	cudaGraphicsGLRegisterBuffer(&cuda_pos_resource, buffer, cudaGraphicsRegisterFlagsNone);
}

void FlipSim::Compute()
{
	eulercompute(this);
}

void FlipSim::EndCompute()
{
	cudaGraphicsUnmapResources(1, &cuda_pos_resource, 0);

	//std::cout << "la taille est " << vit[0].x << std::endl;

}

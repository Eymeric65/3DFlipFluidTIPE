#include "FLIPimpl.h"
#include <assert.h>
#include <iostream>


extern "C" void TrToGr(ParticleSystem * partEngine, FlipSim * flipEngine);

extern "C" void addforces(ParticleSystem * partEngine, FlipSim * flipEngine);

extern "C" void TrToPr(ParticleSystem * partEngine, FlipSim * flipEngine);


FlipSim::FlipSim(float width, float height,float length, float tsize, ParticleSystem partEngine)
{

	BoxSize = make_float3(width, height, length);

	tileSize = tsize;

	partLink = &partEngine;

	timestep = partEngine.TimeStep;

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



	cudaMalloc(&GridSpeed, IndiceCount * sizeof(float3));
	cudaMemset(GridSpeed, 0, IndiceCount * sizeof(float3));

	cudaMalloc(&GridCounter, IndiceCount * sizeof(float));
	cudaMemset(GridCounter, 0, IndiceCount * sizeof(float));

	cudaMalloc(&GridPressure, IndiceCount * sizeof(float3));
	cudaMemset(GridPressure, 0, IndiceCount * sizeof(float3));

	cudaMalloc(&type, IndiceCount * sizeof(unsigned int));



}

void FlipSim::TransferToGrid()
{
	TrToGr(partLink, this);
}

void FlipSim::TransferToParticule()
{
	TrToPr(partLink, this);
}

void FlipSim::AddExternalForces()
{
	addforces(partLink, this);
}

void FlipSim::endSim()
{
	cudaFree(GridSpeed);
	cudaFree(GridCounter);
	cudaFree(type);
	cudaFree(GridPressure);
}

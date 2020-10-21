#include "FLIPimpl.h"
#include <assert.h>
#include <iostream>


extern "C" void TrToGr(ParticleSystem * partEngine, FlipSim * flipEngine);

extern "C" void addforces(ParticleSystem * partEngine, FlipSim * flipEngine);

extern "C" void TrToPr(ParticleSystem * partEngine, FlipSim * flipEngine);

extern "C" void JacobiIter(FlipSim * flipEngine, unsigned int stepNb);

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

	cudaMalloc(&GridPressureB, IndiceCount * sizeof(float3));
	cudaMemset(GridPressureB, 0, IndiceCount * sizeof(float3));

	cudaMalloc(&GridPressureA, IndiceCount * sizeof(float3));

	cudaMalloc(&type, IndiceCount * sizeof(unsigned int));



}

void FlipSim::TransferToGrid()
{
	cudaMemset(GridCounter, 0, IndiceCount * sizeof(float));
	cudaMemset(GridSpeed, 0, IndiceCount * sizeof(float3));
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

void FlipSim::PressureCompute()
{
	JacobiIter(this, 50);
}

void FlipSim::endSim()
{
	cudaFree(GridSpeed);
	cudaFree(GridCounter);
	cudaFree(type);
	cudaFree(GridPressureB);
	cudaFree(GridPressureA);
}

/*
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
*/
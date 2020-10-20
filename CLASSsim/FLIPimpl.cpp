#include "FLIPimpl.h"
#include <assert.h>


extern "C" void TrToGr(ParticleSystem * partEngine, FlipSim * flipEngine);

FlipSim::FlipSim(float width, float height,float length, float tsize, ParticleSystem partEngine)
{

	BoxSize = make_float3(width, height, length);

	tileSize = tsize;

	PartEngine = &partEngine;

	BoxIndice = make_uint3((int)(BoxSize.x / tileSize),
							(int)(BoxSize.y/ tileSize),
							(int)(BoxSize.z / tileSize)  );

	assert( "la taille de la case n'est pas un multiple de la taille de la boite"&&
		    ((BoxSize.x / tileSize) == BoxIndice.x) &&
			((BoxSize.y / tileSize) == BoxIndice.y) &&
			((BoxSize.z / tileSize) == BoxIndice.z)
			);


	IndiceCount = BoxIndice.x * BoxIndice.y * BoxIndice.z;

	cudaMalloc(&GridSpeed, IndiceCount * sizeof(float3));
	cudaMalloc(&GridCounter, IndiceCount * sizeof(float));

	cudaMalloc(&GridPressure, IndiceCount * sizeof(float3));



}

void FlipSim::TransferToGrid()
{
	cudaMemset(GridSpeed, 0, IndiceCount * sizeof(float3));
	cudaMemset(GridCounter, 0, IndiceCount * sizeof(float));

	TrToGr(PartEngine, this);

}

void FlipSim::endSim()
{
	cudaFree(GridSpeed);
	cudaFree(GridCounter);
	cudaFree(GridPressure);
}

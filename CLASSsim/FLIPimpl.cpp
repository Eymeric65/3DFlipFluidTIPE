#include "FLIPimpl.h"

FlipSim::FlipSim(float width, float height,float length, float tsize, ParticleSystem partEngine)
{

	BoxSize = make_float3(width, height, length);

	tileSize = tsize;

	PartEngine = &partEngine;

	BoxIndice = make_float3((int)(BoxSize.x / tileSize),
							(int)(BoxSize.y/ tileSize),
							(int)(BoxSize.z / tileSize)  );

	IndiceCount = BoxIndice.x * BoxIndice.y * BoxIndice.z;

	cudaMalloc(&GridSpeed, IndiceCount * sizeof(float3));

	cudaMalloc(&GridPressure, IndiceCount * sizeof(float3));

}

void FlipSim::TransferToGrid()
{
	int Pcount = PartEngine->PartCount; 

}

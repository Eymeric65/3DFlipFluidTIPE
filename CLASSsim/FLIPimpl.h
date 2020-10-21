#ifndef FLIP_H
#define FLIP_H

#include "ParticleClass.h"

class FlipSim
{


public:

	ParticleSystem* partLink;

	float3 BoxSize;

	float tileSize;

	uint3 BoxIndice;

	int IndiceCount;

	float3* GridSpeed;
	float* GridCounter;

	float3* GridPressure;

	FlipSim(float width,float height,float length,float tsize, ParticleSystem partEngine);

	void TransferToGrid();

	void StartCompute();

	void endSim();

};



#endif
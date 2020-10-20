#ifndef FLIP_H
#define FLIP_H

#include "ParticleClass.h"

class FlipSim
{

	ParticleSystem *PartEngine; 

public:

	float3 BoxSize;

	float tileSize;

	uint3 BoxIndice;

	int IndiceCount;

	float3* GridSpeed;
	float3* GridCounter;

	float3* GridPressure;

	FlipSim(float width,float height,float length,float tsize, ParticleSystem partEngine);

	void TransferToGrid();

	void endSim();

};



#endif
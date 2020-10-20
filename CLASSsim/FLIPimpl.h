#ifndef FLIP_H
#define FLIP_H

#include "ParticleClass.h"

class FlipSim
{

	float3 BoxSize;

	float tileSize;

	float3 BoxIndice;

	int IndiceCount;

	float3* GridSpeed;
	float3* GridPressure;

	ParticleSystem *PartEngine; 

public:

	FlipSim(float width,float height,float length,float tsize, ParticleSystem partEngine);

	void TransferToGrid();

};



#endif
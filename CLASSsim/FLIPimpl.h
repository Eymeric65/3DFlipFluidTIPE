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

	unsigned int* type; // 0 is solid 1 is fluid 2 is air

	float3* GridPressure;

	float timestep;

	FlipSim(float width,float height,float length,float tsize,float tstep, ParticleSystem partEngine);

	void TransferToGrid();

	void AddExternalForces();

	void StartCompute();

	void endSim();

};



#endif
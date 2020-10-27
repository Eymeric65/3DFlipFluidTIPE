#ifndef FLIP_H
#define FLIP_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>


#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

class FlipSim
{

public:

	float3 BoxSize;
	float tileSize;

	uint3 BoxIndice;

	uint3 MACBoxIndice;
	int IndiceCount;
	int MACIndiceCount;

	float3* MACGridSpeed;
	float3* MACGridWeight;

	unsigned int* type; // 0 is solid 1 is fluid 2 is air

	float* GridPressureB;
	float* GridPressureA;

	int PartCount;

	float3* Partpos;
	float3* Partvit;

	float TimeStep;

	struct cudaGraphicsResource* cuda_pos_resource;
	size_t num_bytes_pos;

	FlipSim(float width, float height, float length, float tsize, unsigned int partcount, float tstep);

	void TransferToGrid();

	void TransferToParticule();

	void AddExternalForces();

	void PressureCompute();

	void AddPressure();

	void endSim();

	void StartCompute();

	void Compute();

	void EndCompute();

	void linkPos(GLuint buffer);


};



#endif
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

	uint3 BoxIndice;

	float tileSize;

	int IndiceCount;

	float3* GridSpeed;

	float* GridWeight;

	unsigned int* type; // 0 is solid 1 is fluid 2 is air

	float* GridPressureB;

	float* GridPressureA;

	struct cudaGraphicsResource* cuda_pos_resource;

	float3* Partpos;
	size_t num_bytes_pos;

	int PartCount;

	float3* Partvit;

	float TimeStep;

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
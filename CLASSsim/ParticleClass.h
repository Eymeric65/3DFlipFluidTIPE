#ifndef PARTICLE_H
#define PARTICLE_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>




class ParticleSystem
{
public:

	ParticleSystem(int partcount,float tstep);

private:



public:

	struct cudaGraphicsResource* cuda_pos_resource;

	float3* Partpos;
	size_t num_bytes_pos;

	int PartCount;

	float3* Partvit;

	float TimeStep;

	void StartCompute();

	void Compute();

	void EndCompute();

	void linkPos(GLuint buffer);

	void endSystem();

};

#endif
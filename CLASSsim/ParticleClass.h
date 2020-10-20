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

	ParticleSystem(int partcount);

private:

	struct cudaGraphicsResource* cuda_pos_resource;

	size_t num_bytes_pos;

public:

	int PartCount;

	float3* vit;

	float3* pos;

	void Compute();

	void linkPos(GLuint buffer);

	void endSystem();


};

#endif
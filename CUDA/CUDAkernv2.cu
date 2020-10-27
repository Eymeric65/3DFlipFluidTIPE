#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CLASSsim/FLIPimpl.h"
#include <helper_cuda.h> 

//#define DEBUG
//#define D_VIT
//#define D_TYPE

#define CFL_FORCED

//#define BOUNDARY_WALL_ONLY

__device__ unsigned int gind(unsigned int indiceX, unsigned int indiceY, unsigned int indiceZ, uint3 BoxIndice)
{
	return indiceZ + indiceY * BoxIndice.z + indiceX * BoxIndice.z * BoxIndice.y;
}

__device__ float absmin(float x, float limit)
{
	if (fabsf(x) < fabsf(limit))
	{
		return x;
	}
	else
	{
		return limit* x / fabsf(x);
	}
}

// mettre les vitesses des particules dans la grille
__global__ void TrToGrV2_k(uint3 MACbox,unsigned int partcount,float tsize,float3* MACgridSpeed,float3* MACweight,float3* Ppos,float3* Pvit)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = index; i < partcount; i += stride)
	{
		unsigned int XGridI = (int)(Ppos[index].x / tsize)+1;
		unsigned int YGridI = (int)(Ppos[index].y / tsize)+1;
		unsigned int ZGridI = (int)(Ppos[index].z / tsize)+1;

		//if (YGridI - 1 > 0) { printf("yes"); }

		float ax = Ppos[index].x / tsize - XGridI +1;
		float ay = Ppos[index].y / tsize - YGridI +1;
		float az = Ppos[index].z / tsize - ZGridI +1;

#ifdef DEBUG

		if (ax < 0 || ax>=1 || ay <0 || ay>=1|| az < 0 || az>=1 ) // ne pas oublier de bien centrer les particules
		{
			printf("part : %d mauvais calcul de ax ou ay ou az %f %f %f \n", index,ax,ay,az);
		}

#ifdef D_VIT
		Pvit[index] = make_float3(1, 1, 1);
#endif

#endif

		atomicAdd(&MACgridSpeed[gind(XGridI, YGridI, ZGridI, MACbox)].x, (ax ) * Pvit[index].x);
		atomicAdd(&MACgridSpeed[gind(XGridI, YGridI, ZGridI, MACbox)].y, ( ay ) * Pvit[index].y);
		atomicAdd(&MACgridSpeed[gind(XGridI, YGridI, ZGridI, MACbox)].z,(az) * Pvit[index].z);

		atomicAdd(&MACweight[gind(XGridI, YGridI, ZGridI, MACbox)].x, ax);
		atomicAdd(&MACweight[gind(XGridI, YGridI, ZGridI, MACbox)].y, ay);
		atomicAdd(&MACweight[gind(XGridI, YGridI, ZGridI, MACbox)].z, az);

		atomicAdd(&MACgridSpeed[gind(XGridI-1, YGridI, ZGridI, MACbox)].x, (1-ax) * Pvit[index].x);
		atomicAdd(&MACgridSpeed[gind(XGridI , YGridI-1, ZGridI, MACbox)].y, (1-ay) * Pvit[index].y);
		atomicAdd(&MACgridSpeed[gind(XGridI , YGridI, ZGridI-1, MACbox)].z, (1-az) * Pvit[index].z);

		atomicAdd(&MACweight[gind(XGridI - 1, YGridI, ZGridI, MACbox)].x, (1 - ax) );
		atomicAdd(&MACweight[gind(XGridI, YGridI - 1, ZGridI, MACbox)].y, (1 - ay) );
		atomicAdd(&MACweight[gind(XGridI, YGridI, ZGridI - 1, MACbox)].z, (1 - az) );


	}
}
// apres debug la trilinéarisation est correct !!! 

//fonction copier du dessus qui transfert la vitesse de la grille au particule
__global__ void TrToPrV2_k(uint3 MACbox, unsigned int partcount, float tsize, float3* MACgridSpeed, float3* MACweight, float3* Ppos, float3* Pvit)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = index; i < partcount; i += stride)
	{
		unsigned int XGridI = (int)(Ppos[index].x / tsize) + 1;
		unsigned int YGridI = (int)(Ppos[index].y / tsize) + 1;
		unsigned int ZGridI = (int)(Ppos[index].z / tsize) + 1;


		float ax = Ppos[index].x / tsize - XGridI + 1;
		float ay = Ppos[index].y / tsize - YGridI + 1;
		float az = Ppos[index].z / tsize - ZGridI + 1;

		float xvit = (ax) * MACgridSpeed[gind(XGridI, YGridI, ZGridI, MACbox)].x + (1 - ax) * MACgridSpeed[gind(XGridI - 1, YGridI, ZGridI, MACbox)].x;
		float yvit = (ay) * MACgridSpeed[gind(XGridI, YGridI, ZGridI, MACbox)].y + (1 - ay) * MACgridSpeed[gind(XGridI , YGridI-1, ZGridI, MACbox)].y;
		float zvit = (az) * MACgridSpeed[gind(XGridI, YGridI, ZGridI, MACbox)].z + (1 - az) * MACgridSpeed[gind(XGridI , YGridI, ZGridI-1, MACbox)].z;

		Pvit[index].x = xvit;
		Pvit[index].y = yvit;
		Pvit[index].z = zvit;

#ifdef DEBUG
#ifdef D_VIT
		//if ( (xvit - Pvit[index].x) != 0 || (yvit - Pvit[index].y) !=0|| (zvit - Pvit[index].z)!=0)
		//{
			printf("il y a une difference de : %f %f %f entre la vitesse de base et celle interpole \n", xvit - Pvit[index].x, yvit - Pvit[index].y, zvit - Pvit[index].z);
		//}
#endif
#endif


	}
}
//Apres Debugage l'interpolation est correct !! 


// normaliser la grille
#ifdef DEBUG
__global__ void GridNormalV2_k(uint3 MACbox,float3* MACgridSpeed, float3* MACweight,float* Tweight)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox);

	atomicAdd(Tweight, MACweight[index].x+ MACweight[index].y+ MACweight[index].z);

	MACgridSpeed[index].x = MACgridSpeed[index].x / MACweight[index].x;
	MACgridSpeed[index].y = MACgridSpeed[index].y / MACweight[index].y;
	MACgridSpeed[index].z = MACgridSpeed[index].z / MACweight[index].z;

#ifdef D_VIT

	printf("vitesse de la case %d %d %d est %f %f %f \n", blockIdx.x, blockIdx.y, blockIdx.z, MACgridSpeed[index].x, MACgridSpeed[index].y, MACgridSpeed[index].z);

#endif

}
#else
__global__ void GridNormalV2_k(uint3 MACbox, float3* MACgridSpeed, float3* MACweight)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox);

	MACgridSpeed[index].x = MACgridSpeed[index].x / MACweight[index].x;
	MACgridSpeed[index].y = MACgridSpeed[index].y / MACweight[index].y;
	MACgridSpeed[index].z = MACgridSpeed[index].z / MACweight[index].z;

}
#endif

__global__ void set_typeWater_k(uint3 box,uint3 MACbox, float3* MACweight, unsigned int* type)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, box);


//les bords sont des murs : 


	if (
		MACweight[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].x != 0 ||
		MACweight[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].y != 0 ||
		MACweight[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].z != 0 ||

		MACweight[gind(blockIdx.x+1, blockIdx.y, blockIdx.z, MACbox)].x != 0 ||
		MACweight[gind(blockIdx.x, blockIdx.y+1, blockIdx.z, MACbox)].y != 0 ||
		MACweight[gind(blockIdx.x, blockIdx.y, blockIdx.z+1, MACbox)].z != 0 )
	{
		type[index] = 2;
		if(blockIdx.x>=1)
		{
			type[gind(blockIdx.x - 1, blockIdx.y, blockIdx.z, box)] = 2;
		}
		if (blockIdx.x <= box.x - 2)
		{
			type[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, box)] = 2;
		}
		if (blockIdx.y >= 1)
		{
			type[gind(blockIdx.x , blockIdx.y-1, blockIdx.z, box)] = 2;
		}
		if (blockIdx.y <= box.y - 2)
		{
			type[gind(blockIdx.x , blockIdx.y+1, blockIdx.z, box)] = 2;
		}
		if (blockIdx.z >= 1)
		{
			type[gind(blockIdx.x , blockIdx.y, blockIdx.z-1, box)] = 2;
		}
		if (blockIdx.z <= box.z - 2)
		{
			type[gind(blockIdx.x , blockIdx.y, blockIdx.z+1, box)] = 2;
		}
	}

}

__global__ void set_typeWalls_k(uint3 box, unsigned int* type)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, box);

	if (blockIdx.x == 0 || blockIdx.y == 0 || blockIdx.z == 0 || blockIdx.x == box.x - 1 || blockIdx.y == box.y - 1 || blockIdx.z == box.z - 1)
	{
		type[index] = 1;
	}

#ifdef DEBUG
#ifdef D_TYPE
	if (blockIdx.x== 30&& blockIdx.y==20&& blockIdx.z==20 )
	{
		printf("la case %d %d %d est de type %d \n", blockIdx.x, blockIdx.y, blockIdx.z, type[index]);
	}
#endif
#endif
}

__global__ void add_external_forces_k(uint3 box,uint3 MACbox, float3* MACgridSpeed,unsigned int* type,float tstep)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, box);

	if (type[index] == 2)
	{
		//Gravité
		MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].y -= 9.81 * tstep;
		MACgridSpeed[gind(blockIdx.x , blockIdx.y+1, blockIdx.z, MACbox)].y -= 9.81 * tstep;

	}


}

__global__ void euler_k(int partCount,float3* Ppos,float3* Pvit,float tstep,float tsize)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = index; i < partCount; i += stride)
	{

#ifdef CFL_FORCED

		Ppos[index].x += absmin(Pvit[index].x * tstep, tsize*0.99);
		Ppos[index].y += absmin(Pvit[index].y * tstep, tsize*0.99);
		Ppos[index].z += absmin(Pvit[index].z * tstep, tsize*0.99);

		//printf("hmm %f \n", absmin(Pvit[index].y * tstep, tsize * 0.98));

#else

		Ppos[index].x += Pvit[index].x * tstep;
		Ppos[index].y += Pvit[index].y * tstep;
		Ppos[index].z += Pvit[index].z * tstep;
#endif

	}
}

__global__ void boundaries_k(uint3 box,uint3 MACbox, float3* MACgridSpeed, unsigned int* type)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, box);

	if (type[index] == 1)
	{
		//printf("la case %d %d %d est solide \n", blockIdx.x, blockIdx.y, blockIdx.z);

		if (blockIdx.x >= 1)
		{
			MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].x = 0;
#ifdef BOUNDARY_WALL_ONLY
#else
			MACgridSpeed[gind(blockIdx.x-1, blockIdx.y, blockIdx.z, MACbox)].x = 0;
#endif
		}

		if (blockIdx.y >= 1)
		{
			MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].y = 0;
#ifdef BOUNDARY_WALL_ONLY
#else
			MACgridSpeed[gind(blockIdx.x , blockIdx.y-1, blockIdx.z, MACbox)].y = 0;
#endif
		}

		if (blockIdx.z >= 1)
		{
			MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].z = 0;
#ifdef BOUNDARY_WALL_ONLY
#else
			MACgridSpeed[gind(blockIdx.x , blockIdx.y, blockIdx.z-1, MACbox)].z = 0;
#endif
		}


		if (blockIdx.x <= box.x-2)
		{
			MACgridSpeed[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, MACbox)].x = 0;
#ifdef BOUNDARY_WALL_ONLY
#else
			MACgridSpeed[gind(blockIdx.x + 2, blockIdx.y, blockIdx.z, MACbox)].x = 0;
#endif
		}

		if (blockIdx.y <= box.y - 2)
		{
			MACgridSpeed[gind(blockIdx.x , blockIdx.y+1, blockIdx.z, MACbox)].y = 0;
#ifdef BOUNDARY_WALL_ONLY
#else
			MACgridSpeed[gind(blockIdx.x , blockIdx.y+2, blockIdx.z, MACbox)].y = 0;
#endif
		}

		if (blockIdx.z <= box.z - 2)
		{
			MACgridSpeed[gind(blockIdx.x , blockIdx.y, blockIdx.z+1, MACbox)].z = 0;
#ifdef BOUNDARY_WALL_ONLY
#else
			MACgridSpeed[gind(blockIdx.x , blockIdx.y, blockIdx.z+2, MACbox)].z = 0;
#endif
		}
	}
}

__global__ void pressure_copy_k(uint3 boxind,float* gridpressureB, float* gridpressureA)
{
   unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, boxind);
    gridpressureB[index] = gridpressureA[index];
}

__global__ void jacobi_iter_k(uint3 box, uint3 MACbox, float3* MACgridSpeed, float* gridPressureA , float* gridPressureB,unsigned int* type,float tsize)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, box);

	if (type[index] == 2)
	{

		float div =
			(MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].x - MACgridSpeed[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, MACbox)].x +
			MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].y - MACgridSpeed[gind(blockIdx.x , blockIdx.y + 1, blockIdx.z, MACbox)].y +
			MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].z - MACgridSpeed[gind(blockIdx.x , blockIdx.y, blockIdx.z + 1, MACbox)].z )
			/tsize;

		gridPressureA[index] =
			(gridPressureB[gind(blockIdx.x - 1, blockIdx.y, blockIdx.z, box)] +
				gridPressureB[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, box)] +
				gridPressureB[gind(blockIdx.x, blockIdx.y - 1, blockIdx.z, box)] +
				gridPressureB[gind(blockIdx.x, blockIdx.y + 1, blockIdx.z, box)] +
				gridPressureB[gind(blockIdx.x, blockIdx.y, blockIdx.z - 1, box)] +
				gridPressureB[gind(blockIdx.x, blockIdx.y, blockIdx.z + 1, box)] - div) / 6;

	}

}

__global__ void add_pressure_k(uint3 MACbox, uint3 box, float* gridPressure, float3* MACgridSpeed, float tsize,unsigned int* type)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, box);

	if (type[index] == 2)
	{
		if (type[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, box) != 1])
		{
			MACgridSpeed[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, MACbox)].x -=
				(gridPressure[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, box)] - gridPressure[gind(blockIdx.x, blockIdx.y, blockIdx.z,box)] ) / tsize;
		}

		if (type[gind(blockIdx.x, blockIdx.y +1, blockIdx.z, box) != 1])
		{
			MACgridSpeed[gind(blockIdx.x , blockIdx.y+1, blockIdx.z, MACbox)].y -=
				(gridPressure[gind(blockIdx.x , blockIdx.y +1 , blockIdx.z, box)] - gridPressure[gind(blockIdx.x, blockIdx.y, blockIdx.z, box)]) / tsize;
		}

		if (type[gind(blockIdx.x , blockIdx.y, blockIdx.z +1 , box) != 1])
		{
			MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z +1 , MACbox)].z -=
				(gridPressure[gind(blockIdx.x , blockIdx.y, blockIdx.z +1, box)] - gridPressure[gind(blockIdx.x, blockIdx.y, blockIdx.z, box)]) / tsize;
		}


		if (type[gind(blockIdx.x - 1, blockIdx.y, blockIdx.z, box) != 1])
		{
			MACgridSpeed[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, MACbox)].x -=
				(gridPressure[gind(blockIdx.x , blockIdx.y, blockIdx.z,box)] - gridPressure[gind(blockIdx.x-1, blockIdx.y, blockIdx.z, box)]) / tsize;
		}

		if (type[gind(blockIdx.x, blockIdx.y - 1, blockIdx.z, box) != 1])
		{
			MACgridSpeed[gind(blockIdx.x, blockIdx.y + 1, blockIdx.z, MACbox)].y -=
				(gridPressure[gind(blockIdx.x, blockIdx.y , blockIdx.z, box)] - gridPressure[gind(blockIdx.x, blockIdx.y-1, blockIdx.z, box)]) / tsize;
		}

		if (type[gind(blockIdx.x, blockIdx.y, blockIdx.z -1, box) != 1])
		{
			MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z + 1, MACbox)].z -=
				(gridPressure[gind(blockIdx.x, blockIdx.y, blockIdx.z , box)] - gridPressure[gind(blockIdx.x, blockIdx.y, blockIdx.z-1, box)]) / tsize;
		}

	}
}

extern "C"
void TransfertToGridV2(FlipSim * flipEngine)
{

	unsigned int count = flipEngine->PartCount;

	TrToGrV2_k<<<(count/512+1),512>>>(
		flipEngine->MACBoxIndice,
		flipEngine->PartCount,
		flipEngine->tileSize,
		flipEngine->MACGridSpeed,
		flipEngine->MACGridWeight,
		flipEngine->Partpos,
		flipEngine->Partvit
	);

	getLastCudaError("Kernel execution failed: TrToGrV2_k");

	uint3 MACbox = flipEngine->MACBoxIndice;

	dim3 MACmat(MACbox.x, MACbox.y, MACbox.z);

#ifdef DEBUG

	printf("la grille est : %d %d %d \n", MACbox.x, MACbox.y, MACbox.z);

	float* tweight ;
	cudaMallocManaged(&tweight,sizeof(float));

	GridNormalV2_k << <MACmat, 1 >> > (
		flipEngine->MACBoxIndice,
		flipEngine->MACGridSpeed,
		flipEngine->MACGridWeight,
		tweight
		);

	getLastCudaError("Kernel execution failed: GridNormalV2_k");
	cudaDeviceSynchronize();
	printf("le poid est de : %f \n", *tweight);
#else
	GridNormalV2_k << <MACmat, 1 >> > (
		flipEngine->MACBoxIndice,
		flipEngine->MACGridSpeed,
		flipEngine->MACGridWeight
		);
	getLastCudaError("Kernel execution failed: GridNormalV2_k");
#endif

	uint3 box = flipEngine->BoxIndice;
	dim3 mat(box.x, box.y, box.z);;

	// je pense que ca marche plutot bien ca va mdr on va dire qu'il y a un poil trop de case dite de fluide mais c'est pas tres grave

	set_typeWater_k << <mat, 1 >> > (
		box,
		MACbox,
		flipEngine->MACGridWeight,
		flipEngine->type);

	getLastCudaError("Kernel execution failed: set_typeWater_k");

	set_typeWalls_k << <mat, 1 >> > (
		box,
		flipEngine->type);

	getLastCudaError("Kernel execution failed: set_typeWalls_k");
}


extern "C"
void TransfertToPartV2(FlipSim * flipEngine)
{
	unsigned int count = flipEngine->PartCount;

	TrToPrV2_k << <(count / 512+1), 512 >> > (
		flipEngine->MACBoxIndice,
		flipEngine->PartCount,
		flipEngine->tileSize,
		flipEngine->MACGridSpeed,
		flipEngine->MACGridWeight,
		flipEngine->Partpos,
		flipEngine->Partvit
		);

	getLastCudaError("Kernel execution failed: TrToPrV2_k");

}

extern "C"
void AddExternalForcesV2(FlipSim * flipEngine)
{
	uint3 box = flipEngine->BoxIndice;
	dim3 mat(box.x, box.y, box.z);

	add_external_forces_k<<<mat ,1>>>(
		box, 
		flipEngine->MACBoxIndice,
		flipEngine->MACGridSpeed,
		flipEngine->type,
		flipEngine->TimeStep);

	getLastCudaError("Kernel execution failed: add_external_forces_k");

}

extern "C"
void EulerIntegrateV2(FlipSim * flipEngine)
{
	unsigned int count = flipEngine->PartCount;

	euler_k << <(count / 512+1), 512 >> > (
		count,
		flipEngine->Partpos,
		flipEngine->Partvit,
		flipEngine->TimeStep,
		flipEngine->tileSize);

	getLastCudaError("Kernel execution failed: euler_k");

}

extern "C"
void BoundariesConditionV2(FlipSim * flipEngine)
{
	uint3 box = flipEngine->BoxIndice;
	dim3 mat(box.x, box.y, box.z);

	boundaries_k<<<mat,1>>>(
		box,
		flipEngine->MACBoxIndice,
		flipEngine->MACGridSpeed,
		flipEngine->type);

	getLastCudaError("Kernel execution failed: boundaries_k");
}

extern "C"
void JacobiIterV2(FlipSim * flipEngine,int step)
{
	uint3 box = flipEngine->BoxIndice;
	dim3 mat(box.x, box.y, box.z);

	for (int i = 0; i < step; i++)
	{

		jacobi_iter_k<<<mat,1>>>(
			box,
			flipEngine->MACBoxIndice,
			flipEngine->MACGridSpeed,
			flipEngine->GridPressureA,
			flipEngine->GridPressureB,
			flipEngine-> type,
			flipEngine->tileSize);



		pressure_copy_k<<<mat,1>>>(
			box,
			flipEngine->GridPressureB,
			flipEngine->GridPressureA);

		getLastCudaError("Kernel execution failed: pressure_copy_k");
	}


}

extern "C"
void AddPressureV2(FlipSim * flipEngine)
{
	uint3 box = flipEngine->BoxIndice;
	dim3 mat(box.x, box.y, box.z);

	add_pressure_k<<<mat,1>>>(
		flipEngine->MACBoxIndice,
		box, 
		flipEngine->GridPressureB,
		flipEngine->MACGridSpeed,
		flipEngine->tileSize, 
		flipEngine-> type);

	getLastCudaError("Kernel execution failed: add_pressure_k");
}
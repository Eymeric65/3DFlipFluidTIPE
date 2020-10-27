#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CLASSsim/FLIPimpl.h"
#include <helper_cuda.h> 

//#define DEBUG
//#define D_VIT


__device__ unsigned int gind(unsigned int indiceX, unsigned int indiceY, unsigned int indiceZ, uint3 BoxIndice)
{
	return indiceZ + indiceY * BoxIndice.z + indiceX * BoxIndice.z * BoxIndice.y;
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


// transferer les vitesses dans la grilles
extern "C"
void TransfertToGridV2(FlipSim * flipEngine)
{

	unsigned int count = flipEngine->PartCount;

	TrToGrV2_k<<<(count/512),512>>>(
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

	dim3 numblocks(MACbox.x, MACbox.y, MACbox.z);

#ifdef DEBUG

	printf("la grille est : %d %d %d \n", MACbox.x, MACbox.y, MACbox.z);

	float* tweight ;
	cudaMallocManaged(&tweight,sizeof(float));

	GridNormalV2_k << <numblocks, 1 >> > (
		flipEngine->MACBoxIndice,
		flipEngine->MACGridSpeed,
		flipEngine->MACGridWeight,
		tweight
		);

	getLastCudaError("Kernel execution failed: GridNormalV2_k");
	cudaDeviceSynchronize();
	printf("le poid est de : %f \n", *tweight);
#else
	GridNormalV2_k << <numblocks, 1 >> > (
		flipEngine->MACBoxIndice,
		flipEngine->MACGridSpeed,
		flipEngine->MACGridWeight
		);
	getLastCudaError("Kernel execution failed: GridNormalV2_k");
#endif

}
// DEBUGER le poid des particules est bien correctement stocker

extern "C"
void TransfertToPartV2(FlipSim * flipEngine)
{
	unsigned int count = flipEngine->PartCount;

	TrToPrV2_k << <(count / 512), 512 >> > (
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
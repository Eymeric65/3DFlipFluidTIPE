#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CLASSsim/FLIPimpl.h"
#include <helper_cuda.h> 

__device__ unsigned int getGridInd(unsigned int indiceX, unsigned int indiceY, unsigned int indiceZ, uint3 BoxIndice)
{
    return indiceZ + indiceY * BoxIndice.z + indiceX * BoxIndice.z * BoxIndice.y;
}

__device__ float3 getDiv(unsigned int x, unsigned int y, unsigned int z,float3* Gridspeed, uint3 BoxIndice,float tsize)
{

    if (x > 0 && y > 0 && z > 0)
    {
        unsigned int p = getGridInd(x, y, z, BoxIndice);
        unsigned int xm = getGridInd(x - 1, y, z, BoxIndice);

        unsigned int ym = getGridInd(x, y - 1, z, BoxIndice);

        unsigned int zm = getGridInd(x, y, z - 1, BoxIndice);

        float xr = Gridspeed[p].x - Gridspeed[xm].x;

        float yr = Gridspeed[p].y - Gridspeed[ym].y;

        float zr = Gridspeed[p].z - Gridspeed[zm].z;

        return make_float3(xr / tsize, yr / tsize, zr / tsize);
    }
    else
    {
        return make_float3(0.69, 0.69, 0.69);
    }


}

__device__ int staggeredGrid(float gc, float pc)
{
    if (pc - gc >= 0)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}



__global__ void gridnormal(uint3 boxind, float3* gridspeed, float* gridcounter,unsigned int* type)
{
    unsigned int index = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z,boxind);
    //printf("la cases %d %d %d \n", blockIdx.x, blockIdx.y, blockIdx.z);
    //printf("case %d \n", index);

    if (blockIdx.x == 0 || blockIdx.x == boxind.x-1 ||
        blockIdx.y == 0 || blockIdx.y == boxind.y-1 ||
        blockIdx.z == 0 || blockIdx.z == boxind.z-1)
    {
        type[index] = 0;
    }
    else if (gridcounter[index] != 0)
    {
        gridspeed[index] = make_float3(gridspeed[index].x/ gridcounter[index], gridspeed[index].y/ gridcounter[index], gridspeed[index].z/ gridcounter[index]);
        //printf("la case %d possède des coordonnées x: %f , y: %f , z: %f \n",index, gridspeed[index].x, gridspeed[index].y, gridspeed[index].z);

        type[index] = 1;
    }
    else
    {
        type[index] = 2;
    }

    ///float3 vel = getDiv(blockIdx.x, blockIdx.y, blockIdx.z, gridspeed, boxind, tsize);

    //printf("case %d , %d, %d possede type %d \n", blockIdx.x, blockIdx.y, blockIdx.z,type[index]);
}

__global__ void addforces_k(uint3 boxind, float3* gridspeed, unsigned int* type,float tstep)
{
    unsigned int index = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z, boxind);

    if (type[index] == 1)
    {
        gridspeed[index].y += -9.81 * tstep; 


    }

}

__global__ void boundariescondition(uint3 boxind, float3* gridspeed, unsigned int* type)
{
    unsigned int index = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z, boxind);
    unsigned int indexX = getGridInd(blockIdx.x-1, blockIdx.y, blockIdx.z, boxind);
    unsigned int indexY = getGridInd(blockIdx.x, blockIdx.y-1, blockIdx.z, boxind);
    unsigned int indexZ = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z-1, boxind);

    if (type[index]==0 )
    {
        gridspeed[index].x = 0;
        gridspeed[index].y = 0;
        gridspeed[index].z = 0;
    }

    if ((int)(blockIdx.x ) >= 1 && type[indexX] == 0)
    {
        //printf("lol %d %d \n", (int)(blockIdx.x - 1)>0, (blockIdx.x - 1));
        gridspeed[indexX].x = 0;
    }

    if ((int)(blockIdx.y) >= 1 && type[indexY] == 0)
    {
        gridspeed[indexY].y = 0;
    }

    if ((int)(blockIdx.z) >= 1 && type[indexZ] == 0)
    {
        gridspeed[indexZ].z = 0;
    }
}

__global__ void addpressure_k(uint3 boxind, float3* gridspeed, float* gridpressure, unsigned int* type, float tsize)
{
   unsigned int index = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z, boxind);

   if (type[index] == 1)
   {

       unsigned int indexXm = index - boxind.z * boxind.y;
       unsigned int indexXp = index + boxind.z * boxind.y;


       unsigned int indexYm = index - boxind.z;
       unsigned int indexYp = index + boxind.z;


       unsigned int indexZm = index - 1;
       unsigned int indexZp = index + 1;

       
       if (type[indexXp]!=0)
       {
           gridspeed[index].x -= (gridpressure[indexXp] - gridpressure[index]) / tsize;
       }

       if (type[indexYp] != 0)
       {
           gridspeed[index].y -= (gridpressure[indexYp] - gridpressure[index]) / tsize;
       }

       if (type[indexZp] != 0)
       {
           gridspeed[index].z -= (gridpressure[indexZp] - gridpressure[index]) / tsize;
       }
       

       /*
       gridspeed[index].x -= (gridpressure[indexXp] - gridpressure[index]) / tsize;
       gridspeed[index].y -= (gridpressure[indexYp] - gridspressure[index]) / tsize;
       gridspeed[index].z -= (gridpressure[indexZp] - gridpressure[index]) / tsize;
       */
       float divg = (gridspeed[index].x - gridspeed[indexXm].x + gridspeed[index].y - gridspeed[indexYm].y + gridspeed[index].z - gridspeed[indexZm].z) / tsize;
       if (abs(divg)>1) { printf("la divergence est de : %f \n", divg); }
   }

}

__global__ void JacobiIterationForPressure(uint3 boxind, float3* gridspeed,float* gridpressureB,float* gridpressureA, unsigned int* type,float tsize,float density,float* GridCounter )
{
    unsigned int index = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z, boxind);
    if (type[index] == 1)
    {
        //float3 divg = getDiv(blockIdx.x, blockIdx.y, blockIdx.z, gridspeed, boxind, tsize);
        unsigned int boundCounter = 0;


        unsigned int indexXm = index - boxind.z * boxind.y;
        unsigned int indexXp = index + boxind.z * boxind.y;


        unsigned int indexYm = index - boxind.z;
        unsigned int indexYp = index + boxind.z;


        unsigned int indexZm = index - 1;
        unsigned int indexZp = index + 1;



        float pxm = 0;
        if (blockIdx.x - 1 != 0)
        {
            boundCounter++;
             pxm = gridpressureB[indexXm];
        }
        //printf("ah %f \n", pxm);

        float pxp = 0;
        if (blockIdx.x + 1 != boxind.x-1)
        {
            boundCounter++;
             pxp = gridpressureB[indexXp];
        }

        float pym = 0;
        if (blockIdx.y - 1 != 0)
        {
            boundCounter++;
             pym = gridpressureB[indexYm];
        }

        float pyp = 0;
        if (blockIdx.y + 1 != boxind.y - 1)
        {
            boundCounter++;
             pyp = gridpressureB[indexYp];
        }

        float pzm = 0;
        if (blockIdx.z - 1 != 0)
        {
            boundCounter++;
             pzm = gridpressureB[indexZm];
        }

        float pzp = 0;
        if (blockIdx.z + 1 != boxind.z - 1)
        {
            boundCounter++;
             pzp = gridpressureB[indexZp];
        }
        float divg =
            (gridspeed[index].x - gridspeed[indexXm].x
                +gridspeed[index].y - gridspeed[indexYm].y
                +gridspeed[index].z - gridspeed[indexZm].z ) / tsize;

        /*
        float pxm = 0;
        float dxm = 0;
        if (blockIdx.x > 0)
        {
             pxm = gridpressureB[indexXm];
             dxm = gridspeed[index].x - gridspeed[indexXm].x;
        }
        float pxp = 0;
        if (blockIdx.x < boxind.x-1)
        {
             pxp = gridpressureB[indexXp];
        }
        float pym = 0;
        float dym = 0;
        if (blockIdx.y > 0)
        {
             pym = gridpressureB[indexYm];
             dym = gridspeed[index].y - gridspeed[indexYm].y;
        }
        float pyp = 0;
        if (blockIdx.y < boxind.y - 1)
        {
             pyp = gridpressureB[indexYp];
        }
        float pzm = 0;
        float dzm = 0;
        if (blockIdx.z > 0)
        {
             pzm = gridpressureB[indexZm];
             dzm = gridspeed[index].z - gridspeed[indexZm].z;
        }
        float pzp = 0;
        if (blockIdx.z < boxind.z - 1)
        {
             pzp = gridpressureB[indexZp];
        }

        float divg =
            (dxm
                +  dym
                +  dzm - fmaxf(GridCounter[index] - density, 0)) / tsize;
        */

        gridpressureA[index] = (pxm+pxp+pym+pyp+pzm+pzp - divg)/ 6;
        


        if (index == 1003)
        {
            //printf("il y a %f \n", fmaxf(GridCounter[index] - density, 0));
           //printf("la div est %f %f %f \n", divgx, divgy, divgz);
        }

    }
}

__global__ void PressureCopy(uint3 boxind,float* gridpressureB, float* gridpressureA)
{
    unsigned int index = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z, boxind);
    gridpressureB[index] = gridpressureA[index];
}

__global__ void TransferToGrid(unsigned int partcount, float3* pos, float3* vit, float3* gridspeed, float* gridcounter, float tsize, uint3 BoxIndice)
{

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < partcount; i += stride)
    {

        //printf("index is %d \n", index);

        int gridindexX = (int)(pos[index].x / tsize);
        int gridindexY = (int)(pos[index].y / tsize);
        int gridindexZ = (int)(pos[index].z / tsize);


                    ///vit[index] = make_float3(1, 1, 1); test purpose only
            //printf("la case %d \n", gridind);

            float centerposX = gridindexX * tsize;
            float centerposY = gridindexY * tsize;
            float centerposZ = gridindexZ * tsize;

            float stX = staggeredGrid(centerposX, pos[index].x);
            float stY = staggeredGrid(centerposY, pos[index].y);
            float stZ = staggeredGrid(centerposZ, pos[index].z);

            unsigned int gridindX = getGridInd(gridindexX + stX, gridindexY, gridindexZ, BoxIndice);
            unsigned int gridindY = getGridInd(gridindexX, gridindexY + stY, gridindexZ, BoxIndice);
            unsigned int gridindZ = getGridInd(gridindexX, gridindexY, gridindexZ + stZ, BoxIndice);


            unsigned int gridindXi = getGridInd(gridindexX - 1 - stX, gridindexY, gridindexZ, BoxIndice);
            unsigned int gridindYi = getGridInd(gridindexX, gridindexY - 1 - stY, gridindexZ, BoxIndice);
            unsigned int gridindZi = getGridInd(gridindexX, gridindexY, gridindexZ - 1 - stZ, BoxIndice);

            float ax = fabsf((pos[index].x - centerposX) / 2 * tsize + 0.5);
            float ay = fabsf((pos[index].y - centerposY) / 2 * tsize + 0.5);
            float az = fabsf((pos[index].z - centerposZ) / 2 * tsize + 0.5);

            unsigned int gridind = getGridInd(gridindexX, gridindexY, gridindexZ, BoxIndice);

            atomicAdd(&gridspeed[gridindX].x, vit[index].x*ax);
            atomicAdd(&gridspeed[gridindY].y, vit[index].y*ay);
            atomicAdd(&gridspeed[gridindZ].z, vit[index].z*az);

            atomicAdd(&gridspeed[gridindXi].x, vit[index].x * (1-ax));
            atomicAdd(&gridspeed[gridindYi].y, vit[index].y * (1-ay));
            atomicAdd(&gridspeed[gridindZi].z, vit[index].z * (1-az));

            atomicAdd(&gridcounter[gridindX],  ax);
            atomicAdd(&gridcounter[gridindY],  ay);
            atomicAdd(&gridcounter[gridindZ],  az);

            atomicAdd(&gridcounter[gridindXi], (1 - ax));
            atomicAdd(&gridcounter[gridindYi],  (1 - ay));
            atomicAdd(&gridcounter[gridindZi],  (1 - az));
            /*
            if (index == 1999)
            {
                printf("atomic test %f , %f \n", vit[index].y, gridcounter[gridind]);
            }*/

        
        //printf("indice de la particule dans la grille %d %d \n ",index, getGridInd(gridindexX, gridindexY, gridindexZ, BoxIndice));

        //pos[index] = make_float3(3, 3, 3);s

        //printf("la particule %d possède des coordonnées x: %f , y: %f , z: %f \n",index, pos[index].x, pos[index].y, pos[index].z);

    }


}

// ce transfert de vitesse est vraiment plutot brutal méthode PIC
__global__ void TransfertToParticule(unsigned int partcount, float3* pos, float3* vit, float3* gridspeed, float* gridcounter, float tsize, uint3 BoxIndice)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < partcount; i += stride)
    {

        //printf("index is %d \n", index);

        int gridindexX = (int)(pos[index].x / tsize);
        int gridindexY = (int)(pos[index].y / tsize);
        int gridindexZ = (int)(pos[index].z / tsize);

            //printf("la case %d \n", gridind);

            float centerposX = gridindexX * tsize;
            float centerposY = gridindexY * tsize;
            float centerposZ = gridindexZ * tsize;

            float stX = staggeredGrid(centerposX, pos[index].x);
            float stY = staggeredGrid(centerposY, pos[index].y);
            float stZ = staggeredGrid(centerposZ, pos[index].z);

            unsigned int gridindX = getGridInd(gridindexX + stX, gridindexY, gridindexZ, BoxIndice);
            unsigned int gridindY = getGridInd(gridindexX, gridindexY + stY, gridindexZ, BoxIndice);
            unsigned int gridindZ = getGridInd(gridindexX, gridindexY, gridindexZ + stZ, BoxIndice);

            
            unsigned int gridindXi = getGridInd(gridindexX - 1-stX, gridindexY, gridindexZ, BoxIndice);
            unsigned int gridindYi = getGridInd(gridindexX, gridindexY -1-stY, gridindexZ, BoxIndice);
            unsigned int gridindZi = getGridInd(gridindexX, gridindexY, gridindexZ -1- stZ, BoxIndice);
            
            float ax = fabsf((pos[index].x - centerposX)/2*tsize+0.5);
            float ay = fabsf((pos[index].y - centerposY) / 2 * tsize+0.5);
            float az = fabsf((pos[index].z - centerposZ) / 2 * tsize+0.5);


            
            vit[index] = make_float3(gridspeed[gridindX].x*ax + (1-ax)* gridspeed[gridindXi].x
                , gridspeed[gridindY].y * ay + (1 - ay) * gridspeed[gridindYi].y
                , gridspeed[gridindZ].z * az + (1 - az) * gridspeed[gridindZi].z); // direct assigmement
            
            /*
            vit[index] = make_float3(gridspeed[gridindX].x 
                , gridspeed[gridindY].y 
                , gridspeed[gridindZ].z ); // direct assigmement
            */
            //printf("la vitesse de la particule %d est %f %f %f \n", index, vit[index].x, vit[index].y, vit[index].z);

        

        //printf("indice de la particule dans la grille %d %d \n ",index, getGridInd(gridindexX, gridindexY, gridindexZ, BoxIndice));

        //pos[index] = make_float3(3, 3, 3);

        //printf("la particule %d possède des coordonnées x: %f , y: %f , z: %f \n",index, pos[index].x, pos[index].y, pos[index].z);

    }

}

__global__ void EulerIntegration(unsigned int partcount, float3* pos, float3* vit, float tstep, float3 boxsize, float tsize)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < partcount; i += stride)
    {

        pos[index].x += vit[index].x * tstep ;
        pos[index].y +=vit[index].y * tstep ;
        pos[index].z += vit[index].z * tstep ;

        if (pos[index].x < tsize)
        {
            pos[index].x = tsize * 1.01;
            vit[index].x = 0;
        }
        if (pos[index].x > boxsize.x - tsize)
        {
            pos[index].x = boxsize.x - tsize * 1.01;
            vit[index].x = 0;
        }

        if (pos[index].y < tsize)
        {
            pos[index].y = tsize * 1.01;
            vit[index].y = 0;
        }
        if (pos[index].y > boxsize.y - tsize)
        {
            pos[index].y = boxsize.y - tsize * 1.01;
            vit[index].y = 0;
        }

        if (pos[index].z < tsize)
        {
            pos[index].z = tsize * 1.01;
            vit[index].z = 0;
        }
        if (pos[index].z > boxsize.z - tsize)
        {
            pos[index].z = boxsize.z - tsize * 1.01;
            vit[index].z = 0;
        }
    }
}

__global__ void checking(uint3 boxind, float3* gridspeed,unsigned int* type )
{
    unsigned int index = getGridInd(blockIdx.x, blockIdx.y, blockIdx.z, boxind);
    if (blockIdx.y == 0)
    {

        //printf("vitesse %f \n", gridspeed[index].y);
        //printf("type %d \n", type[index]);
    }

}

extern "C"
void eulercompute(FlipSim * flipEngine)
{
    EulerIntegration << <1000, 1024 >> > (
        flipEngine->PartCount,
        flipEngine->Partpos,
        flipEngine->Partvit,
        flipEngine->TimeStep,
        flipEngine->BoxSize,
        flipEngine->tileSize);

    getLastCudaError("Kernel execution failed: EulerIntegration");

    uint3 box = flipEngine->BoxIndice;

    dim3 numblocks(box.x, box.y, box.z);

    checking << <numblocks, 1 >> > (
        box,
        flipEngine->GridSpeed,
        flipEngine->type);

    getLastCudaError("Kernel execution failed: checker");




}
// -------------------------------------------------------------------------------------------
extern "C"
void TrToGr( FlipSim * flipEngine)
{
    //std::cout <<"nombre de particule "<< partEngine->PartCount << std::endl;

    TransferToGrid << <200, 512 >> > (
        flipEngine->PartCount,
        flipEngine->Partpos,
        flipEngine->Partvit,
        flipEngine->GridSpeed,
        flipEngine->GridWeight,
        flipEngine->tileSize,
        flipEngine->BoxIndice);

    getLastCudaError("Kernel execution failed: TransferToGrid");

    uint3 box = flipEngine->BoxIndice;
    
    dim3 numblocks(box.x, box.y, box.z);
    
    gridnormal << <numblocks, 1 >> > (
        box,
        flipEngine->GridSpeed,
        flipEngine->GridWeight,
        flipEngine->type);

    getLastCudaError("Kernel execution failed: gridnormal");
}

extern "C"
void addforces( FlipSim * flipEngine)
{
    uint3 box = flipEngine->BoxIndice;

    dim3 numblocks(box.x, box.y, box.z);

    addforces_k << <numblocks, 1 >> > (
        box,
        flipEngine->GridSpeed,
        flipEngine->type,
        flipEngine->TimeStep);

    getLastCudaError("Kernel execution failed: addforces_k");

    
    boundariescondition << <numblocks, 1 >> > (
        box,
        flipEngine->GridSpeed,
        flipEngine->type);

    getLastCudaError("Kernel execution failed: boundariescondition");

    



}

extern "C"
void TrToPr(FlipSim * flipEngine)
{
    TransfertToParticule << <200, 512 >> > (
        flipEngine->PartCount,
        flipEngine->Partpos,
        flipEngine->Partvit,
        flipEngine->GridSpeed,
        flipEngine->GridWeight,
        flipEngine->tileSize,
        flipEngine->BoxIndice);

    getLastCudaError("Kernel execution failed: TransfertToParticule");

}

extern "C"
void JacobiIter( FlipSim * flipEngine, unsigned int stepNb)
{

    //cudaMemset(flipEngine->GridPressureB, 0, flipEngine->IndiceCount * sizeof(float3));

    uint3 box = flipEngine->BoxIndice;

    dim3 numblocks(box.x, box.y, box.z);

    for (int i = 0; i < stepNb; i += 1)
    {
        
        JacobiIterationForPressure << <numblocks, 1 >> > (
            box,
            flipEngine->GridSpeed,
            flipEngine->GridPressureB,
            flipEngine->GridPressureA,
            flipEngine->type,
            flipEngine->tileSize,
            8.0,
            flipEngine->GridWeight);

        getLastCudaError("Kernel execution failed: JacobiIterationForPressure");
            
        PressureCopy<<<numblocks,1>>>(
            box, 
            flipEngine->GridPressureB, 
            flipEngine->GridPressureA);

        getLastCudaError("Kernel execution failed: PressureCopy");
    }

}

extern "C"
void AddPressureForce(FlipSim * flipEngine)
{
    uint3 box = flipEngine->BoxIndice;

    dim3 numblocks(box.x, box.y, box.z);

    addpressure_k<<<numblocks ,1>>>(
        box,
        flipEngine->GridSpeed, 
        flipEngine->GridPressureA, 
        flipEngine->type,
        flipEngine->tileSize);

    getLastCudaError("Kernel execution failed: addpressure_k");

}

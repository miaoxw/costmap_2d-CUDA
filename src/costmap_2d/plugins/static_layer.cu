#include <stdio.h>
#include <costmap_2d/cuda_static_layer.h>

#define TPB 512
#define NO_INFORMATION 255

__global__ void updateWithTrueOverwriteKernel(unsigned char *master,unsigned char *costmap,unsigned long size, int min_i, int min_j, int max_i, int max_j, int span)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	int deltaj=id/(max_j-min_j);	//Row
	int deltai=id%(max_i-min_i);	//Coloum
	int j=min_j+deltaj;
	int i=min_i+deltai;
	int index=span*j+i;
	if(index<size)
		master[index]=costmap[index];
}

__global__ void updateWithMaxKernel(unsigned char *master,unsigned char *costmap,unsigned long size, int min_i, int min_j, int max_i, int max_j, int span)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	int deltaj=id/(max_j-min_j);	//Row
	int deltai=id%(max_i-min_i);	//Coloum
	int j=min_j+deltaj;
	int i=min_i+deltai;
	int index=span*j+i;
	if(index<size)
	{
		if(costmap[i]==NO_INFORMATION)
			return;
		unsigned char oldCost=master[index];
		if(oldCost==NO_INFORMATION||oldCost<costmap[index])
			master[index]=costmap[index];
	}
}

void costmap_2d::cuda::updateWithTrueOverwrite(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j, unsigned char *costmap_)
{
	//Unnecessary. Has been checked outside.
	/*********
	if (!enabled_)
    	return;
	 *********/
	unsigned char *master = master_grid.getCharMap();
	unsigned int span = master_grid.getSizeInCellsX();
	unsigned long size=master_grid.getSizeInCellsX()*master_grid.getSizeInCellsY();

	unsigned char *cuda_master=NULL;
	unsigned char *cuda_costmap=NULL;
	cudaMalloc(&cuda_master,sizeof(unsigned char)*size);
	cudaMalloc(&cuda_costmap,sizeof(unsigned char)*size);

	cudaMemcpy(cuda_master,master,sizeof(unsigned char)*size,cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_costmap,costmap_,sizeof(unsigned char)*size,cudaMemcpyHostToDevice);

	updateWithTrueOverwriteKernel<<<(size+TPB-1)/TPB,TPB>>>(cuda_master,cuda_costmap,size,min_i,min_j,max_i,max_j,span);

	cudaMemcpy(master,cuda_master,sizeof(unsigned char)*size,cudaMemcpyDeviceToHost);
	cudaFree(cuda_master);
	cudaFree(cuda_costmap);
}

void costmap_2d::cuda::updateWithMax(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j, unsigned char *costmap_)
{
	//Unnecessary. Has been checked outside.
	/*********
	if (!enabled_)
    	return;
	 *********/
	unsigned char *master = master_grid.getCharMap();
	unsigned int span = master_grid.getSizeInCellsX();
	unsigned long size=master_grid.getSizeInCellsX()*master_grid.getSizeInCellsY();

	unsigned long sizeToUpdate=(max_j-min_j)*(max_i-min_i);

	unsigned char *cuda_master=NULL;
	unsigned char *cuda_costmap=NULL;
	cudaMalloc(&cuda_master,sizeof(unsigned char)*size);
	cudaMalloc(&cuda_costmap,sizeof(unsigned char)*size);

	cudaMemcpy(cuda_master,master,sizeof(unsigned char)*size,cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_costmap,costmap_,sizeof(unsigned char)*size,cudaMemcpyHostToDevice);

	updateWithMaxKernel<<<(sizeToUpdate+TPB-1)/TPB,TPB>>>(cuda_master,cuda_costmap,size,min_i,min_j,max_i,max_j,span);

	cudaMemcpy(master,cuda_master,sizeof(unsigned char)*size,cudaMemcpyDeviceToHost);
	cudaFree(cuda_master);
	cudaFree(cuda_costmap);
}
                  
#include <costmap_2d/cuda_costmap_2d.h>
#include <costmap_2d/cuda_static_layer.h>

#define TPB 512
#define NO_INFORMATION 255

__global__ void updateWithTrueOverwriteKernel(unsigned char *master,unsigned char *costmap,unsigned long size, int min_x, int min_y, int max_x, int max_y, int span)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	int deltay=id/(max_x-min_x);
	int deltax=id%(max_x-min_x);
	int x=min_x+deltax;
	int y=min_y+deltay;
	int index=span*y+x;
	if(index<size)
		master[index]=costmap[index];
}

__global__ void updateWithMaxKernel(unsigned char *master,unsigned char *costmap,unsigned long size, int min_x, int min_y, int max_x, int max_y, int span)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	int deltay=id/(max_x-min_x);
	int deltax=id%(max_x-min_x);
	int x=min_x+deltax;
	int y=min_y+deltay;
	int index=span*y+x;
	if(index<size)
	{
		if(costmap[index]==NO_INFORMATION)
			return;
		unsigned char oldCost=master[index];
		if(oldCost==NO_INFORMATION||oldCost<costmap[index])
			master[index]=costmap[index];
	}
}

void costmap_2d::cuda::updateWithTrueOverwrite(costmap_2d::Costmap2D& master_grid, int min_x, int min_y, int max_x, int max_y, unsigned char *costmap_)
{
	//Unnecessary. Has been checked outside.
	/*********
	if (!enabled_)
    	return;
	 *********/
	unsigned char *master = master_grid.getCharMap();
	unsigned int span = master_grid.getSizeInCellsX();
	unsigned long size=master_grid.getSizeInCellsX()*master_grid.getSizeInCellsY();

	unsigned long sizeToUpdate=(max_y-min_y)*(max_x-min_x);

	unsigned char *cuda_master=NULL;
	unsigned char *cuda_costmap=NULL;
	cudaMalloc(&cuda_master,sizeof(unsigned char)*size);
	cudaMalloc(&cuda_costmap,sizeof(unsigned char)*size);

	cudaMemcpy(cuda_master,master,sizeof(unsigned char)*size,cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_costmap,costmap_,sizeof(unsigned char)*size,cudaMemcpyHostToDevice);

	updateWithTrueOverwriteKernel<<<(sizeToUpdate+TPB-1)/TPB,TPB>>>(cuda_master,cuda_costmap,size,min_x,min_y,max_x,max_y,span);

	cudaMemcpy(master,cuda_master,sizeof(unsigned char)*size,cudaMemcpyDeviceToHost);
	cudaFree(cuda_master);
	cudaFree(cuda_costmap);
}

void costmap_2d::cuda::updateWithMax(costmap_2d::Costmap2D& master_grid, int min_x, int min_y, int max_x, int max_y, unsigned char *costmap_)
{
	//Unnecessary. Has been checked outside.
	/*********
	if (!enabled_)
    	return;
	 *********/
	unsigned char *master = master_grid.getCharMap();
	unsigned int span = master_grid.getSizeInCellsX();
	unsigned long size=master_grid.getSizeInCellsX()*master_grid.getSizeInCellsY();

	unsigned long sizeToUpdate=(max_y-min_y)*(max_x-min_x);

	unsigned char *cuda_master=NULL;
	unsigned char *cuda_costmap=NULL;
	cudaMalloc(&cuda_master,sizeof(unsigned char)*size);
	cudaMalloc(&cuda_costmap,sizeof(unsigned char)*size);

	cudaMemcpy(cuda_master,master,sizeof(unsigned char)*size,cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_costmap,costmap_,sizeof(unsigned char)*size,cudaMemcpyHostToDevice);

	updateWithMaxKernel<<<(sizeToUpdate+TPB-1)/TPB,TPB>>>(cuda_master,cuda_costmap,size,min_x,min_y,max_x,max_y,span);

	cudaMemcpy(master,cuda_master,sizeof(unsigned char)*size,cudaMemcpyDeviceToHost);
	cudaFree(cuda_master);
	cudaFree(cuda_costmap);
}

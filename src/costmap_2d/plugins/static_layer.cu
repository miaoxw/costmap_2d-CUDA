#include <stdio.h>
#include <costmap_2d/cuda_static_layer.h>

__global__ void hello()
{
	printf("Hello CUDA from threadIdx=%d\n",threadIdx.x);
}

void costmap_2d::cuda::updateWithTrueOverwrite(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j)
{
	//Unnecessary. Has been checked outside.
	/*********
	if (!enabled_)
    	return;
	 *********/
	unsigned char *master = master_grid.getCharMap();
	unsigned int span = master_grid.getSizeInCellsX();
	unsigned long size=master_grid.getSizeInCellsX()*master_grid.getSizeInCellsY();
	
	for (int j = min_j; j < max_j; j++)
	{
		unsigned int it = span*j+min_i;
		for (int i = min_i; i < max_i; i++)
    	{
      		master[it] = costmap_[it];
      		it++;
    	}
  	}
	hello<<<1,4>>>();
	cudaDeviceSynchronize();
}
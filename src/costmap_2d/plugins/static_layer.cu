#include <tf/tf.h>
#include <costmap_2d/cuda_static_layer.h>

#define TPB 512
#define NO_INFORMATION 255

struct CostMapParameters
{
	double origin_x;
	double origin_y;
	double resolution;
	int span;
};

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

__device__ bool worldToMap(double origin_x,double origin_y,double resolution,double wx,double wy,unsigned int &mx,unsigned int &my)
{
	if(wx<origin_x||wy<origin_y)
		return false;
	
	mx=(int)((wx-origin_x)/resolution);
	my=(int)((wy-origin_y)/resolution);
	return true;

	//Missing down parts of test statement is placed at the if-clause in the large if.
}

__global__ void rollingUpdateCostsKernel(unsigned char *master, CostMapParameters masterParams,
	unsigned char *costmap, CostMapParameters staticLayerParams, CostMapParameters layeredCostmapParams,
	tf::TransformData serializedTF,	int min_x, int min_y, int max_x, int max_y, bool use_maximum)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	int deltay=id/(max_x-min_x);
	int deltax=id%(max_x-min_x);
	int x=min_x+deltax;
	int y=min_y+deltay;

	double wx,wy;
	wx=layeredCostmapParams.origin_x+(x+0.5)*layeredCostmapParams.resolution;
	wy=layeredCostmapParams.origin_y+(y+0.5)*layeredCostmapParams.resolution;
	double new_wx,new_wy;
	new_wx=serializedTF.m_basis.m_el[0].m_floats[0]*wx+serializedTF.m_basis.m_el[0].m_floats[1]*wy+serializedTF.m_origin.m_floats[0];
	new_wy=serializedTF.m_basis.m_el[1].m_floats[0]*wx+serializedTF.m_basis.m_el[1].m_floats[1]*wy+serializedTF.m_origin.m_floats[1];

	unsigned int mx,my;
	if(worldToMap(staticLayerParams.origin_x,staticLayerParams.origin_y,
	staticLayerParams.resolution,new_wx,new_wy,mx,my))
	{
		int master_index=i*masterParams.span+j;
		int costmap_index=mx*staticLayerParams.span+my;
		if(costmap_index<sizeof(costmap)/sizeof(costmap[0]))
		{
			if(use_maximum)
				master[master_index]=
					master[master_index]>costmap[costmap_index]?master[master_index]:costmap[costmap_index];
			else
				master[master_index]=costmap[costmap_index];
		}
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
 
void costmap_2d::cuda::static_layer::rollingUpdateCosts(costmap_2d::Costmap2D& master_grid, tf::StampedTransform tf, costmap_2d::Costmap2D *staticLayer_costmap, costmap_2d::Costmap2D *layered_costmap, bool use_maximum, int min_x, int min_y, int max_x, int max_y)
{
	struct tf::TransformData serializedTF;
	tf.serialize(serializedTF);

	unsigned char *master = master_grid.getCharMap();
	unsigned char *costmap_grid = staticLayer_costmap->getCharMap();

	struct CostMapParameters masterParams,staticLayerParams,layeredCostmapParams;	
	masterParams.span = master_grid.getSizeInCellsX();
	masterParams.resolution = master_grid.getResolution();
	masterParams.origin_x = master_grid.getOriginX();
	masterParams.origin_y = master_grid.getOriginY();
	unsigned long master_size = master_grid.getSizeInCellsX()*master_grid.getSizeInCellsY();
	staticLayerParams.span = staticLayer_costmap->getSizeInCellsX();
	staticLayerParams.resolution = staticLayer_costmap->getResolution();
	staticLayerParams.origin_x = staticLayer_costmap->getOriginX();
	staticLayerParams.origin_y = staticLayer_costmap->getOriginY();
	unsigned long staticLayerCostmap_size=staticLayer_costmap->getSizeInCellsX()*staticLayer_costmap->getSizeInCellsY();
	layeredCostmapParams.span = layered_costmap->getSizeInCellsX();
	layeredCostmapParams.resolution = layered_costmap->getResolution();
	layeredCostmapParams.origin_x = layered_costmap->getOriginX();
	layeredCostmapParams.origin_y = layered_costmap->getOriginY();

	unsigned long sizeToUpdate=(max_y-min_y)*(max_x-min_x);

	unsigned char *cuda_master=NULL;
	unsigned char *cuda_staticLayerCostmap=NULL;
	cudaMalloc(&cuda_master,sizeof(unsigned char)*master_size);
	cudaMalloc(&cuda_staticLayerCostmap,sizeof(unsigned char)*staticLayerCostmap_size);

	cudaMemcpy(cuda_master,master,sizeof(unsigned char)*master_size,cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_staticLayerCostmap,costmap_grid,sizeof(unsigned char)*staticLayerCostmap_size,cudaMemcpyHostToDevice);

	rollingUpdateCostsKernel<<<(sizeToUpdate+TPB-1)/TPB,TPB>>>(cuda_master,master_size,masterParams,cuda_staticLayerCostmap,staticLayerCostmap_size,staticLayerParams,layeredCostmapParams,serializedTF,min_x,min_y,max_x,max_y,use_maximum);

	cudaMemcpy(master,cuda_master,sizeof(unsigned char)*master_size,cudaMemcpyDeviceToHost);
	cudaFree(cuda_master);
	cudaFree(cuda_staticLayerCostmap);
}

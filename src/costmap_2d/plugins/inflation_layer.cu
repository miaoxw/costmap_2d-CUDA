#include <costmap_2d/cuda_inflation_layer.h>
#include <cstring>
#include <vector>

#define TPB 512

using std::memcpy;
using std::vector;
using costmap_2d::CellData;

__global__ void setCostFloodingInflateUnkown(unsigned char *master, unsigned long master_size_x, unsigned long master_size_y, unsigned char *cachedCost_1D, unsigned int cacheSize, CellData *obstaclesArray, unsigned int obstaclesArray_count, unsigned int inflation_radius)
{

}

__global__ void setCostFloodingNoInflateUnkown(unsigned char *master, unsigned long master_size_x, unsigned long master_size_y, unsigned char *cachedCost_1D, unsigned int cacheSize, CellData *obstaclesArray, unsigned int obstaclesArray_count, unsigned int inflation_radius)
{
    
}

void costmap_2d::cuda::inflation_layer::setCostFlooding(unsigned char *master, unsigned long master_size_x, unsigned long master_size_y, unsigned char **cached_cost, const vector<CellData> &obstacles, unsigned int inflation_radius, bool inflate_unknown)
{
    unsigned int cacheSize=inflation_radius+2;

    if(obstacles.empty())
        return;
    
    //Compress the original 2D cached cost into 1D for more convenient cuda_memcpy
    unsigned char *cachedCost_1D=new unsigned char[cacheSize*cacheSize];
    for(int i=0;i<cacheSize;i++)
        memcpy(cachedCost_1D+i*cacheSize,cached_cost[i],cacheSize);
    
    CellData *obstaclesArray=new CellData[obstacles.size()];
    memcpy(obstaclesArray,&obstacles[0],obstacles.size());

    unsigned char *cuda_master;
    unsigned char *cuda_cachedCost_1D;
    CellData *cuda_obstaclesArray;
    cudaMalloc(&cuda_master,sizeof(unsigned char)*master_size_x*master_size_y);
    cudaMalloc(&cuda_cachedCost_1D,sizeof(unsigned char)*cacheSize*cacheSize);
    cudaMalloc(&cuda_obstaclesArray,sizeof(CellData)*obstacles.size());
    
    cudaMemcpy(cuda_master,master,sizeof(unsigned char)*master_size_x*master_size_y,cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_cachedCost_1D,cachedCost_1D,sizeof(unsigned char)*cacheSize*cacheSize,cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_obstaclesArray,obstaclesArray,sizeof(CellData)*obstacles.size(),cudaMemcpyHostToDevice);

    //2*inflation_radius+1 is actually inflation diameter, but we still had better pass raduis into kernel.
    unsigned long totalWorkload=obstacles.size()*(2*inflation_radius+1)*(2*inflation_radius+1);
    if(inflate_unknown)
        setCostFloodingInflateUnkown<<<(totalWorkload+TPB-1)/TPB,TPB>>>(cuda_master,master_size_x,master_size_y,cuda_cachedCost_1D,cacheSize,cuda_obstaclesArray,obstacles.size(),inflation_radius);
    else
        setCostFloodingNoInflateUnkown<<<(totalWorkload+TPB-1)/TPB,TPB>>>(cuda_master,master_size_x,master_size_y,cuda_cachedCost_1D,cacheSize,cuda_obstaclesArray,obstacles.size(),inflation_radius);
    
    cudaMemcpy(master,cuda_master,sizeof(unsigned char)*master_size_x*master_size_y,cudaMemcpyDeviceToHost);
    cudaFree(cuda_master);
    cudaFree(cuda_cachedCost_1D);
    cudaFree(cuda_obstaclesArray);
    delete [] cachedCost_1D;
    delete [] obstaclesArray;
}
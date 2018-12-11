#include <costmap_2d/obstacle_layer.h>
#include <costmap_2d/cuda_obstacle_layer.h>
#include <costmap_2d/observation.h>

#include <cmath>
#include <vector>

using std::vector;

#define TPB 512

using std::min;
using std::max;
using std::ceil;
// original worldToMap
__device__ bool worldToMap(double wx, double wy, unsigned int& mx, unsigned int& my, double origin_x, double origin_y,
    double resolution, unsigned int size_x, unsigned int size_y)
{
  if (wx < origin_x || wy < origin_y)
    return false;

  mx = (int)((wx - origin_x) / resolution);
  my = (int)((wy - origin_y) / resolution);

  if (mx < size_x && my < size_y)
    return true;

  return false;
}

// __device__ bool worldToMap(double origin_x,double origin_y,double resolution,double wx,double wy,unsigned int &mx,unsigned int &my)
// {
// 	if(wx<origin_x||wy<origin_y)
// 		return false;
	
// 	mx=(int)((wx-origin_x)/resolution);
// 	my=(int)((wy-origin_y)/resolution);
// 	return true;

// 	//Missing down parts of test statement is placed at the if-clause in the large if.
// }


/*
void bresenham2D(unsigned char *costmap, unsigned char value, unsigned int abs_da, unsigned int abs_db, int error_b, int offset_a,
    int offset_b, unsigned int offset, unsigned int max_length)
{
    unsigned int end = std::min(max_length, abs_da);
    //TODO 
    // for (unsigned int i = 0; i < end; ++i)
    // {
    //     costmap[offset]=value;
    //     offset += offset_a;
    //     error_b += abs_db;
    //     if ((unsigned int)error_b >= abs_da)
    //     {
    //         offset += offset_b;
    //         error_b -= abs_da;
    //     }
    // }
    costmap[offset]=value;
}
*/
__device__ void bresenham2D(
    unsigned char *costmap, char value, 
    int abs_da,  int abs_db, int error_b,
    int offset_a,int offset_b, 
    int offset,  int max_length)
{
    // unsigned int end = std::min(max_length, abs_da);
    unsigned int end = (max_length< abs_da)?max_length:abs_da;

    //TODO 
    for (unsigned int i = 0; i < end; ++i)
    {
        costmap[offset]=value;
        offset += offset_a;
        error_b += abs_db;
        if ((unsigned int)error_b >= abs_da)
        {
            offset += offset_b;
            error_b -= abs_da;
        }
    }
    costmap[offset]=value;
}


/*
void raytraceLine(unsigned char *costmap, unsigned char value, unsigned int x0, unsigned int y0,
    unsigned int x1, unsigned int y1, unsigned int max_length, unsigned int size_x)
{
    int dx = x1 - x0;
    int dy = y1 - y0;

    unsigned int abs_dx = abs(dx);
    unsigned int abs_dy = abs(dy);

    int offset_dx = dx>0?1:-1;
    int offset_dy = (dy>0?1:-1) * size_x;

    unsigned int offset = y0 * size_x + x0;

    // we need to chose how much to scale our dominant dimension, based on the maximum length of the line
    double dist = hypot((double)dx, (double)dy);
    double scale = (dist == 0.0) ? 1.0 : std::min(1.0, max_length / dist);

    // if x is dominant
    if (abs_dx >= abs_dy)
    {
        int error_y = abs_dx / 2;
        bresenham2D(costmap, value, abs_dx, abs_dy, error_y, offset_dx, offset_dy, offset, (unsigned int)(scale * abs_dx));
        return;
    }

    // otherwise y is dominant
    int error_x = abs_dy / 2;
    bresenham2D(costmap, value, abs_dy, abs_dx, error_x, offset_dy, offset_dx, offset, (unsigned int)(scale * abs_dy));
}
*/
// __global__ void raytraceLine(unsigned char *costmap, unsigned char value, unsigned int x0, unsigned int y0,
//     unsigned int x1, unsigned int y1, unsigned int max_length, unsigned int size_x)
// {
//     int dx = x1 - x0;
//     int dy = y1 - y0;

//     unsigned int abs_dx = abs(dx);
//     unsigned int abs_dy = abs(dy);

//     int offset_dx = dx>0?1:-1;
//     int offset_dy = (dy>0?1:-1) * size_x;

//     unsigned int offset = y0 * size_x + x0;

//     // we need to chose how much to scale our dominant dimension, based on the maximum length of the line
//     double dist = hypot((double)dx, (double)dy);
//     double scale = (dist == 0.0) ? 1.0 : std::min(1.0, max_length / dist);

//     // if x is dominant
//     if (abs_dx >= abs_dy)
//     {
//         int error_y = abs_dx / 2;
//         bresenham2D(costmap, value, abs_dx, abs_dy, error_y, offset_dx, offset_dy, offset, (unsigned int)(scale * abs_dx));
//         return;
//     }

//     // otherwise y is dominant
//     int error_x = abs_dy / 2;
//     bresenham2D(costmap, value, abs_dy, abs_dx, error_x, offset_dy, offset_dx, offset, (unsigned int)(scale * abs_dy));
// }


/* original
void updateRaytraceBounds(double ox, double oy, double wx, double wy, double range,
    double* min_x, double* min_y, double* max_x, double* max_y)
{
    double dx = wx-ox, dy = wy-oy;
    double full_distance = hypot(dx, dy);
    double scale = std::min(1.0, range / full_distance);
    double ex = ox + dx * scale, ey = oy + dy * scale;
    *min_x=min(ex,*min_x);
    *min_y=min(ey,*min_y);
    *max_x=max(ex,*max_x);
    *max_y=max(ey,*max_y);
}
*/
__global__ void updateRaytraceBounds(double ox, double oy, double wx, double wy, double range,
    double* min_x, double* min_y, double* max_x, double* max_y)
{
   
}

__global__ void rayTraceFreeSpaceKernel(
    unsigned char *obstacleLayerCostmap,int pointLen,
    costmap_2d::cuda::obstacle_layer::PointXY *cloudPoints,
    double raytrace_range,unsigned char defaultValue, 
    double origin_x, double origin_y, 
    double ox, double oy, 
    double map_end_x, double map_end_y, 
    double resolution,
    unsigned int size_x, unsigned int size_y,
    unsigned int x0, unsigned int y0, 
    double *min_x, double *min_y,
    double *max_x, double *max_y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id>=pointLen){
        return;
    }
    double wx = cloudPoints[id].x;
    double wy = cloudPoints[id].y;

    double a = wx - ox;
    double b = wy - oy;


    // the minimum value to raytrace from is the origin
    if (wx < origin_x)
    {
        double t = (origin_x - ox) / a;
        wx = origin_x;
        wy = oy + b * t;
    }
    if (wy < origin_y)
    {
        double t = (origin_y - oy) / b;
        wx = ox + a * t;
        wy = origin_y;
    }

    // the maximum value to raytrace to is the end of the map
    if (wx > map_end_x)
    {
        double t = (map_end_x - ox) / a;
        wx = map_end_x - .001;
        wy = oy + b * t;
    }
    if (wy > map_end_y)
    {
        double t = (map_end_y - oy) / b;
        wx = ox + a * t;
        wy = map_end_y - .001;
    }

    // now that the vector is scaled correctly... we'll get the map coordinates of its endpoint
    unsigned int x1, y1;

    // check for legality just in case
    if (!worldToMap(wx, wy, x1, y1,origin_x,origin_y,resolution,size_x,size_y)){
        
    }else{
        //unsigned int cell_raytrace_range = (unsigned int)max(0.0,ceil(raytrace_range/resolution));
        unsigned int cell_raytrace_range = (unsigned int)max(0.0,ceil(raytrace_range/resolution));

        // and finally... we can execute our trace to clear obstacles along that line
        //raytraceLine(obstacleLayerCostmap, defaultValue, x0, y0, x1, y1, cell_raytrace_range, size_x);
        int dx = x1 - x0;
        int dy = y1 - y0;
    
        unsigned int abs_dx = abs(dx);
        unsigned int abs_dy = abs(dy);
    
        int offset_dx = dx>0?1:-1;
        int offset_dy = (dy>0?1:-1) * size_x;
    
        unsigned int offset = y0 * size_x + x0;
    
        // we need to chose how much to scale our dominant dimension, based on the maximum length of the line
        double dist = hypot((double)dx, (double)dy);
        

       // double scale = (dist == 0.0) ? 1.0 : std::min(1.0, raytrace_range / dist);
        double scale = (dist == 0.0) ? 1.0 : (1.0<         (raytrace_range / dist) ?     1.0:    (raytrace_range / dist)  )  ;

        // if x is dominant
        if (abs_dx >= abs_dy)
        {
            int error_y = abs_dx / 2;
            bresenham2D(obstacleLayerCostmap, defaultValue, abs_dx, abs_dy, error_y, offset_dx, offset_dy, offset, (unsigned int)(scale * abs_dx));
            return;
        }
    
        // otherwise y is dominant
        int error_x = abs_dy / 2;
        bresenham2D(obstacleLayerCostmap, defaultValue, abs_dy, abs_dx, error_x, offset_dy, offset_dx, offset, (unsigned int)(scale * abs_dy));
    
    
        
        
        
        //updateRaytraceBounds(ox, oy, wx, wy, clearing_observation.raytrace_range_, min_x, min_y, max_x, max_y);
        double dx1 = wx-ox, dy1 = wy-oy;
        double full_distance = hypot(dx1, dy1);
        //double scale1 = std::min(1.0, raytrace_range / full_distance);

        double scale1 = 1.0 < (raytrace_range / full_distance) ?  1.0: (raytrace_range / full_distance);


        double ex = ox + dx1 * scale, ey = oy + dy1 * scale1;
        *min_x=min(ex,*min_x);
        *min_y=min(ey,*min_y);
        *max_x=max(ex,*max_x);
        *max_y=max(ey,*max_y);
    }
      

    


/*
    // for each point in the cloud, we want to trace a line from the origin and clear obstacles along it
    for (unsigned int i = 0; i < cloud.points.size(); ++i)
    {
        double wx = cloud.points[i].x;
        double wy = cloud.points[i].y;
            
        // now we also need to make sure that the enpoint we're raytracing
        // to isn't off the costmap and scale if necessary
        double a = wx - ox;
        double b = wy - oy;
    
        // the minimum value to raytrace from is the origin
        if (wx < origin_x)
        {
            double t = (origin_x - ox) / a;
            wx = origin_x;
            wy = oy + b * t;
        }
        if (wy < origin_y)
        {
            double t = (origin_y - oy) / b;
            wx = ox + a * t;
            wy = origin_y;
        }
    
        // the maximum value to raytrace to is the end of the map
        if (wx > map_end_x)
        {
            double t = (map_end_x - ox) / a;
            wx = map_end_x - .001;
            wy = oy + b * t;
        }
        if (wy > map_end_y)
        {
            double t = (map_end_y - oy) / b;
            wx = ox + a * t;
            wy = map_end_y - .001;
        }
    
        // now that the vector is scaled correctly... we'll get the map coordinates of its endpoint
        unsigned int x1, y1;
    
        // check for legality just in case
        if (!worldToMap(wx, wy, x1, y1,origin_x,origin_y,resolution,size_x,size_y))
            continue; 
    
        unsigned int cell_raytrace_range = (unsigned int)max(0.0,ceil(clearing_observation.raytrace_range_/resolution));
        
        // and finally... we can execute our trace to clear obstacles along that line
        raytraceLine(costmap, defaultValue, x0, y0, x1, y1, cell_raytrace_range, size_x);
    
    }  
*/


}


void costmap_2d::cuda::obstacle_layer::rayTraceFreeSpace(
    unsigned char *costmap, int sizeInCellsX,int sizeInCellsY, unsigned char defaultValue, 
    const Observation& clearing_observation,costmap_2d::cuda::obstacle_layer::PointXY *cloudPoints,int pointLen,
     double origin_x, double origin_y,
     double ox, double oy,
    double map_end_x, double map_end_y, 
     double resolution,
      unsigned int size_x, unsigned int size_y,
       unsigned int x0, unsigned int y0, 
       double *min_x, double *min_y,
       double *max_x, double *max_y)
{
    //store cloudPoints
    costmap_2d::cuda::obstacle_layer::PointXY *in_cloudPoints=NULL;

    //int pointLen=( sizeof(cloudPoints) / sizeof(double) )/ ( sizeof(cloudPoints[0]) / sizeof(double) );
    
    unsigned char *obstacleLayerCostmap=NULL;
    unsigned long obstacleLayerCostmap_size=sizeInCellsX*sizeInCellsY;
	
    double *in_min_x=NULL;
    double *in_min_y=NULL;
    double *in_max_x=NULL;
    double *in_max_y=NULL;
    

   
    cudaMalloc(&in_cloudPoints,sizeof(costmap_2d::cuda::obstacle_layer::PointXY)*pointLen);
    cudaMalloc(&obstacleLayerCostmap,sizeof(unsigned char)*obstacleLayerCostmap_size);
    cudaMalloc(&in_min_x,1*sizeof(double));
    cudaMalloc(&in_min_y,1*sizeof(double));
    cudaMalloc(&in_max_x,1*sizeof(double));
    cudaMalloc(&in_max_y,1*sizeof(double));

    cudaMemcpy(in_min_x,min_x,1*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(in_min_y,min_y,1*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(in_max_x,max_x,1*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(in_max_y,max_y,1*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(in_cloudPoints,cloudPoints,sizeof(costmap_2d::cuda::obstacle_layer::PointXY)*pointLen,cudaMemcpyHostToDevice);
    cudaMemcpy(obstacleLayerCostmap,costmap,sizeof(unsigned char)*obstacleLayerCostmap_size,cudaMemcpyHostToDevice);


    rayTraceFreeSpaceKernel<<<(pointLen+TPB-1)/TPB,TPB>>>
    (obstacleLayerCostmap,pointLen,in_cloudPoints,
        clearing_observation.raytrace_range_,defaultValue,
        origin_x,origin_y,ox,oy,
        map_end_x,map_end_y,resolution,
        size_x,size_y,x0,y0,
        in_min_x,in_min_y,in_max_x,in_max_y);
    
    cudaMemcpy(costmap,obstacleLayerCostmap,sizeof(costmap_2d::cuda::obstacle_layer::PointXY)*pointLen,cudaMemcpyDeviceToHost);
    cudaMemcpy(min_x,in_min_x,1*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(min_y,in_min_y,1*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(max_x,in_max_x,1*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(max_y,in_max_y,1*sizeof(double),cudaMemcpyDeviceToHost);

    cudaFree(in_cloudPoints);
    cudaFree(in_min_x);
    cudaFree(in_min_y);
    cudaFree(in_max_x);
    cudaFree(in_max_y);
    cudaFree(obstacleLayerCostmap);



   
     
}
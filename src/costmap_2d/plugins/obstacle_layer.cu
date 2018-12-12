#include <costmap_2d/obstacle_layer.h>
#include <costmap_2d/cuda_obstacle_layer.h>
#include <costmap_2d/observation.h>

#include <pcl/impl/point_types.hpp>

#include <cmath>

#define TPB 256

using std::max;
using std::ceil;

using pcl::PointXYZ;
using costmap_2d::cuda::obstacle_layer::MyPointXY;

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

__device__ void bresenham2D(unsigned char *costmap, unsigned char value, unsigned int abs_da, unsigned int abs_db, int error_b, int offset_a,
    int offset_b, unsigned int offset, unsigned int max_length)
{
    unsigned int end = max_length<abs_da?max_length:abs_da;
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

__device__ void raytraceLine(unsigned char *costmap, unsigned char value, unsigned int x0, unsigned int y0,
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
    double scale = (dist == 0.0) ? 1.0 : (1.0<max_length/dist?1.0:max_length/dist);

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

__device__ void updateRaytraceBounds(double ox, double oy, double wx, double wy, double range,
    double* min_x, double* min_y, double* max_x, double* max_y)
{
    double dx = wx-ox, dy = wy-oy;
    double full_distance = hypot(dx, dy);
    double scale = 1.0<range/full_distance?1.0:range/full_distance;
    double ex = ox + dx * scale, ey = oy + dy * scale;
    *min_x=min(ex,*min_x);
    *min_y=min(ey,*min_y);
    *max_x=max(ex,*max_x);
    *max_y=max(ey,*max_y);
}

__global__ void rayTraceFreeSpaceKernel(unsigned char *costmap, unsigned char defaultValue, double raytraceRange,
    MyPointXY *cloudArray, int cloudArray_size, double origin_x, double origin_y, double ox, double oy,
    double map_end_x, double map_end_y, double resolution, unsigned int size_x, unsigned int size_y,
    unsigned int x0, unsigned int y0, double *min_x, double *min_y, double *max_x, double *max_y)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=cloudArray_size)
        return;

    double wx=cloudArray[id].x;
    double wy=cloudArray[id].y;

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
        return;

    unsigned int cell_raytrace_range = (unsigned int)max(0.0,ceil(raytraceRange/resolution));
    
    // and finally... we can execute our trace to clear obstacles along that line
    raytraceLine(costmap, defaultValue, x0, y0, x1, y1, cell_raytrace_range, size_x);

    updateRaytraceBounds(ox, oy, wx, wy, raytraceRange, min_x, min_y, max_x, max_y);
}

void costmap_2d::cuda::obstacle_layer::rayTraceFreeSpace(unsigned char *costmap, unsigned char defaultValue, double raytraceRange,
    MyPointXY *cloudArray, unsigned int cloudArraySize, double ox, double oy, double origin_x, double origin_y,
    double map_end_x, double map_end_y, double resolution, unsigned int size_x, unsigned int size_y,
    unsigned int x0, unsigned int y0, double *min_x, double *min_y, double *max_x, double *max_y)
{
    MyPointXY *cuda_cloudArray;
    cudaMalloc(&cuda_cloudArray,sizeof(MyPointXY)*cloudArraySize);
    cudaMemcpy(cuda_cloudArray,cloudArray,sizeof(MyPointXY)*cloudArraySize,cudaMemcpyHostToDevice);
    unsigned char *cuda_costmap;
    cudaMalloc(&cuda_costmap,sizeof(unsigned char)*size_x*size_y);
    cudaMemcpy(cuda_costmap,costmap,sizeof(unsigned char)*size_x*size_y,cudaMemcpyHostToDevice);
    double *cuda_min_x,*cuda_min_y,*cuda_max_x,*cuda_max_y;
    cudaMalloc(&cuda_min_x,sizeof(double));
    cudaMemcpy(cuda_min_x,min_x,sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_min_y,sizeof(double));
    cudaMemcpy(cuda_min_y,min_y,sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_max_x,sizeof(double));
    cudaMemcpy(cuda_max_x,max_x,sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_max_y,sizeof(double));
    cudaMemcpy(cuda_max_y,max_y,sizeof(double),cudaMemcpyHostToDevice);

    rayTraceFreeSpaceKernel<<<(cloudArraySize+TPB-1)/TPB,TPB>>>(cuda_costmap,defaultValue,raytraceRange,cuda_cloudArray,cloudArraySize,origin_x,origin_y,ox,oy,map_end_x,map_end_y,resolution,size_x,size_y,x0,y0,cuda_min_x,cuda_min_y,cuda_max_x,cuda_max_y);

    cudaMemcpy(costmap,cuda_costmap,sizeof(unsigned char)*size_x*size_y,cudaMemcpyDeviceToHost);
    cudaMemcpy(min_x,cuda_min_x,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(min_y,cuda_min_y,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(max_x,cuda_max_x,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(max_y,cuda_max_y,sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(cuda_cloudArray);
    cudaFree(cuda_costmap);
    cudaFree(cuda_min_x);
    cudaFree(cuda_min_y);
    cudaFree(cuda_max_x);
    cudaFree(cuda_max_y);
}
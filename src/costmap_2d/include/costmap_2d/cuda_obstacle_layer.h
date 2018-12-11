#include <costmap_2d/obstacle_layer.h>

#ifndef CUDA_OBSTACLE_LAYER_H
#define CUDA_OBSTACLE_LAYER_H
namespace costmap_2d
{
    namespace cuda
    {
        namespace obstacle_layer
        {
            struct PointXY
            {
                double x,y;
            };

            void rayTraceFreeSpace(unsigned char *costmap, int sizeInCellsX,int sizeInCellsY,
                unsigned char defaultValue, 
                const Observation& clearing_observation,
                PointXY *cloudPoints,int lenth,
                double origin_x, double origin_y, 
                double ox,double oy,double map_end_x,double map_end_y, 
                double resolution, 
                unsigned int size_x, unsigned int size_y, 
                unsigned int x0, unsigned int y0, 
                double *min_x, double *min_y, double *max_x, double *may_y);
        }
    }
}
#endif

#include <costmap_2d/obstacle_layer.h>

#ifndef CUDA_OBSTACLE_LAYER_H
#define CUDA_OBSTACLE_LAYER_H
namespace costmap_2d
{
    namespace cuda
    {
        namespace obstacle_layer
        {
            void rayTraceFreeSpace(const Observation& clearing_observation, double origin_x, double origin_y, double map_end_x, double map_end_y, double resolution);
        }
    }
}
#endif

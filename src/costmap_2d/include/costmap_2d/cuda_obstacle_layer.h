#include <costmap_2d/obstacle_layer.h>

#ifndef CUDA_OBSTACLE_LAYER_H
#define CUDA_OBSTACLE_LAYER_H
namespace costmap_2d
{
    namespace cuda
    {
        namespace obstcale_layer
        {
            void rayTraceLine(costmap_2d::Costmap2D::Markcell action, unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1, unsigned int max_length = UINT_MAX);
        }
    }
}
#endif

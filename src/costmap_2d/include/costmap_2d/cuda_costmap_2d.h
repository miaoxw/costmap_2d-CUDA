#include <costmap_2d/static_layer.h>

#ifndef CUDA_COSTMAP2D_H
#define CUDA_COSTMAP2D_H
namespace costmap_2d
{
    namespace cuda
    {
        void updateWithTrueOverwrite(costmap_2d::Costmap2D& master_grid, int min_x, int min_y, int max_x, int max_y, unsigned char *costmap_);
        void updateWithMax(costmap_2d::Costmap2D& master_grid, int min_x, int min_y, int max_x, int max_y, unsigned char *costmap_);
    }
}
#endif

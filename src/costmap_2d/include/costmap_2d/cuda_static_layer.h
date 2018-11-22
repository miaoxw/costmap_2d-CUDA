#include <costmap_2d/static_layer.h>

#ifndef CUDA_STATIC_LAYER_H
#define CUDA_STATIC_LAYER_H
namespace costmap_2d
{
    namespace cuda
    {
        void updateWithTrueOverwrite(costmap_2d::Costmap2D& master_grid, int min_x, int min_y, int max_x, int max_y, unsigned char *costmap_);
        void updateWithMax(costmap_2d::Costmap2D& master_grid, int min_x, int min_y, int max_x, int max_y, unsigned char *costmap_);
        namespace static_layer
        {
            void rollingUpdateCosts(costmap_2d::Costmap2D& master_grid, tf::StampedTransform tf, costmap_2d::Costmap2D *staticLayer_costmap, costmap_2d::Costmap2D *layered_costmap, bool use_maximum, int min_x, int min_y, int max_x, int max_y);
        }
    }
}
#endif

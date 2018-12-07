#include <costmap_2d/inflation_layer.h>

#ifndef CUDA_INFLATION_LAYER_H
#define CUDA_INFLATION_LAYER_H
namespace costmap_2d
{
	namespace cuda
	{
		namespace inflation_layer
		{
			void setCostFlooding(unsigned char *master, unsigned long master_size_x, unsigned long master_size_y, unsigned char **cached_cost, const std::vector<costmap_2d::CellData> &obstacles, unsigned int inflation_radius, bool inflate_unknown);
		}
	} // namespace cuda
} // namespace costmap_2d
#endif
#include <costmap_2d/cuda_inflation_layer.h>
#include <vector>

#define TPB 512

using std::vector;
using costmap_2d::CellData;

void costmap_2d::cuda::inflation_layer::setCostFlooding(unsigned char *master, unsigned char **cached_cost, const vector<CellData> &obstacles, unsigned int inflation_radius, bool inflate_unknown)
{

}
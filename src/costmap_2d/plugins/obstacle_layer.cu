#include <costmap_2d/obstacle_layer.h>
#include <costmap_2d/cuda_obstacle_layer.h>
#include <costmap_2d/observation.h>

#define TPB 512

void costmap_2d::cuda::obstacle_layer::rayTraceFreeSpace(const Observation& clearing_observation, double origin_x, double origin_y, double map_end_x, double map_end_y, double resolution)
{

}
#include <vector>
#include <math.h>
#include <assert.h>

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Core>

#include <pcl/io/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/centroid.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/random_sample.h>
#include <random>
#include <chrono>
#include <iostream>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

#include "registration.h"

class simu_process{

public:
    simu_process();
    ~simu_process();
    void model_data_generation(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud_ori, int N, float tho, Eigen::Matrix4f T, float noise_bound,
                                    cloudPoints &src_points, cloudPoints &tgt_points, std::vector<bool> &inlier_mask);

    void simu_data_generation(int N, float tho, Eigen::Matrix4f T, float noise_bound,
                                   cloudPoints &src_points, cloudPoints &tgt_points, std::vector<bool> &inlier_mask);

protected:

    void norm_pointcloud(cloudPoints &cloud);

    void addNoiseAndOutliers(cloudPoints &tgt_cloud, int N_OUTLIERS, std::vector<bool> &inlier_mask, float noise_bound);


};

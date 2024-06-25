#include <iostream>
#include <Eigen/Core>
#include <chrono>
#include <vector>

#include <algorithm>
#include <random>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

//external libraries
#include <pcl/io/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>


#include "simu_process.h"
#include "registration.h"

inline float getAngularError(Eigen::Matrix3f R_exp, Eigen::Matrix3f R_est) {
    return std::abs(std::acos(fmin(fmax(((R_exp.transpose() * R_est).trace() - 1) / 2, -1.0), 1.0)) * 180 / M_PI);
}

int main(){

    srand(unsigned(time(NULL)));

    // read original point cloud
    std::string ply_path = "../models/bun_zipper.ply";
//    std::string ply_path = "../models/Armadillo.ply";
    pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud_ori(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile(ply_path, *src_cloud_ori);

    // max N = 35947 for bunny and 172974 for Armadillo
    int N = 10000;                          // correspondence number
    float tho = 0.95;                       // outlier ratio
    float noise_bound = 0.05;              // maximum length of noise vector

    // generate random Transformation
    Eigen::Matrix4f T = Eigen::Matrix4f::Zero();

    T << 0.803058, -0.195693, -0.562852, -0.384886,
            -0.282914, 0.706093, -0.649147, -1.38517,
            0.524459, 0.680542, 0.511669, -2.86804,
            0, 0, 0, 1;

    Eigen::Matrix3f Rotation = T.block<3,3>(0, 0);
    Eigen::Vector3f Translation = T.block<3,1>(0, 3);

    // data generation
    cloudPoints src_points;          // 3*N
    cloudPoints tgt_points;          // 3*N
    std::vector<bool> inlier_mask;   // true: inlier   false: outlier

    simu_process gen_data;
    gen_data.model_data_generation(src_cloud_ori, N, tho, T, noise_bound, src_points, tgt_points, inlier_mask);


    // visualization
    pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < N; ++i) {
        src_cloud->push_back(
                pcl::PointXYZ(static_cast<float>(src_points.col(i)[0]), static_cast<float>(src_points.col(i)[1]),
                              static_cast<float>(src_points.col(i)[2])));
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < N; ++i) {
        tgt_cloud->push_back(
                pcl::PointXYZ(static_cast<float>(tgt_points.col(i)[0]), static_cast<float>(tgt_points.col(i)[1]),
                              static_cast<float>(tgt_points.col(i)[2])));
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_cloud_inlier(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < N; ++i) {
        if(inlier_mask[i]){
            tgt_cloud_inlier->push_back(
                    pcl::PointXYZ(static_cast<float>(tgt_points.col(i)[0]), static_cast<float>(tgt_points.col(i)[1]),
                                  static_cast<float>(tgt_points.col(i)[2])));
        }
    }

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("test"));
    viewer->setBackgroundColor(255, 255, 255);
    viewer->setCameraPosition(-7.088909,0.489558,6.150008,-0.408311,-1.214921,-2.231068,0.290319,0.956217,0.036946);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_color(src_cloud, 255, 180, 0);
    viewer->addPointCloud(src_cloud, src_color, "src");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.5, "src");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> dst_color(tgt_cloud, 0, 166, 237);
    viewer->addPointCloud(tgt_cloud, dst_color, "dst");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, "dst");

    viewer->addPointCloud(tgt_cloud_inlier, dst_color, "dst_inlier");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "dst_inlier");


    // visualize the correspondences
//    for (size_t j = 0; j < src_cloud->size(); ++j)
//    {
//        std::stringstream ss_line;
//        ss_line << "correspondence_line_" << j;
//        pcl::PointXYZ & src_keypoint = src_cloud->points[j];
//        pcl::PointXYZ & tgt_keypoint = tgt_cloud->points[j];
//
//        if(inlier_mask[j]){
//            viewer->addLine<pcl::PointXYZ, pcl::PointXYZ> (src_keypoint, tgt_keypoint, 0, 255, 0, ss_line.str ());
//            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 0.01, ss_line.str ());
//        }
//        else{
//            viewer->addLine<pcl::PointXYZ, pcl::PointXYZ> (src_keypoint, tgt_keypoint, 255, 0, 0, ss_line.str ());
//            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 0.01, ss_line.str ());
//        }
//    }

    viewer->spin();
    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }



    std::chrono::steady_clock::time_point TEAR_begin = std::chrono::steady_clock::now();
    TEAR::registration regis_TEAR;
    regis_TEAR.noise_bound = noise_bound;
    regis_TEAR.BRANCH_ACCURACY = 5e-2;
    regis_TEAR.TEAR_est(src_points, tgt_points);
    std::chrono::steady_clock::time_point TEAR_end = std::chrono::steady_clock::now();

    Eigen::Matrix3f rotation_TEAR = regis_TEAR.est_rotation.cast<float>();
    Eigen::Vector3f translation_TEAR = regis_TEAR.est_translation.cast<float>();
    Eigen::Matrix4f T_TEAR;
    T_TEAR.block<3,3>(0, 0) = rotation_TEAR;
    T_TEAR.block<3,1>(0, 3) = translation_TEAR;

    float R_error_TEAR = getAngularError(Rotation, rotation_TEAR);
    float t_error_TEAR = (Translation - translation_TEAR).norm();
    float time_cost_TEAR =
            std::chrono::duration_cast<std::chrono::microseconds>(TEAR_end - TEAR_begin).count() / 1000000.0;

    std::cout << "Rotation error(TEAR): " << R_error_TEAR << std::endl;
    std::cout << "Translation error(TEAR): " << t_error_TEAR << std::endl;
    std::cout << "Time cost(TEAR): " << time_cost_TEAR << std::endl;


    pcl::visualization::PCLVisualizer::Ptr viewer_final(new pcl::visualization::PCLVisualizer("test"));
    viewer_final->setBackgroundColor(255, 255, 255);
    viewer_final->setCameraPosition(-7.088909,0.489558,6.150008,-0.408311,-1.214921,-2.231068,0.290319,0.956217,0.036946);

    pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*src_cloud, *src_cloud_trans, T_TEAR);

    viewer_final->addPointCloud(src_cloud_trans, src_color, "src_mesh");
    viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, "src_mesh");

    viewer_final->addPointCloud(tgt_cloud, dst_color, "dst_mesh");
    viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, "dst_mesh");

    viewer_final->addPointCloud(tgt_cloud_inlier, dst_color, "dst_inlier");
    viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "dst_inlier");



    while (!viewer_final->wasStopped()) {
        viewer_final->spinOnce();
    }

    return 0;
}


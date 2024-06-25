#include "../include/simu_process.h"

simu_process::simu_process() = default;
simu_process::~simu_process() = default;


void simu_process::norm_pointcloud(cloudPoints &points) {

    Eigen::Vector3f center = points.rowwise().mean();
    // subtract mean
    Eigen::Matrix3Xf points_m = points.colwise() - center;

    float scale_max = points_m.colwise().norm().maxCoeff();

    points = points_m/scale_max;

}

void simu_process::addNoiseAndOutliers(cloudPoints &tgt, int N_OUTLIERS, std::vector<bool> &inlier_mask, float noise_bound) {
    // Add uniform noise
    Eigen::Matrix<float, 3, Eigen::Dynamic> noise =
            Eigen::Matrix<float, 3, Eigen::Dynamic>::Random(3, tgt.cols()) * noise_bound;

    tgt = tgt + noise;

    // Add outliers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis2(0, tgt.cols() - 1); // pos of outliers
    for (int i = 0; i < N_OUTLIERS; ++i) {

        int c_outlier_idx;
        do {
            c_outlier_idx = dis2(gen);
        }while(c_outlier_idx >= inlier_mask.size() || !inlier_mask[c_outlier_idx]);

        Eigen::Matrix<float, 3, 1> rand_vector = Eigen::Matrix<float, 3, 1>::Random();
        tgt.col(c_outlier_idx) = 10*rand_vector;
        inlier_mask[c_outlier_idx] = false;
    }
}


void simu_process::model_data_generation(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud_ori, int N, float tho, Eigen::Matrix4f T, float noise_bound,
                                              cloudPoints &src_points, cloudPoints &tgt_points, std::vector<bool> &inlier_mask){

    // downsample
    pcl::RandomSample<pcl::PointXYZ> down_sample(true);
    down_sample.setInputCloud(src_cloud_ori);
    down_sample.setSample(N);
    down_sample.setSeed(random());
    down_sample.filter(*src_cloud_ori);

    // to cloudPoints
    src_points.resize(3, N);
    for (int i = 0; i < N; ++i) {
        src_points.col(i) << src_cloud_ori->points[i].x, src_cloud_ori->points[i].y, src_cloud_ori->points[i].z;
    }
    norm_pointcloud(src_points);

    // tgt points generation
    Eigen::Matrix<float, 4, Eigen::Dynamic> src_homo;
    src_homo.resize(4, src_points.cols());
    src_homo.topRows(3) = src_points;
    src_homo.bottomRows(1) = Eigen::Matrix<float, 1, Eigen::Dynamic>::Ones(N);
    Eigen::Matrix<float, 4, Eigen::Dynamic> tgt_homo = T * src_homo;

    tgt_points.resize(3, N);
    tgt_points = tgt_homo.topRows(3);
    inlier_mask = std::vector<bool>(tgt_points.cols(), true);
    addNoiseAndOutliers(tgt_points, static_cast<int>(N * tho), inlier_mask, noise_bound);

}



void simu_process::simu_data_generation(int N, float tho, Eigen::Matrix4f T, float noise_bound,
                                             cloudPoints &src_points, cloudPoints &tgt_points, std::vector<bool> &inlier_mask){

    src_points = Eigen::Matrix<float, 3, Eigen::Dynamic>::Random(3, N).array().abs();

    // tgt points generation
    Eigen::Matrix<float, 4, Eigen::Dynamic> src_homo;
    src_homo.resize(4, src_points.cols());
    src_homo.topRows(3) = src_points;
    src_homo.bottomRows(1) = Eigen::Matrix<float, 1, Eigen::Dynamic>::Ones(N);
    Eigen::Matrix<float, 4, Eigen::Dynamic> tgt_homo = T * src_homo;

    tgt_points.resize(3, N);
    tgt_points = tgt_homo.topRows(3);
    inlier_mask = std::vector<bool>(tgt_points.cols(), true);
    addNoiseAndOutliers(tgt_points, static_cast<int>(N * tho), inlier_mask, noise_bound);

}
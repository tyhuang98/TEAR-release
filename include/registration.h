#ifndef REGISTRATION
#define REGISTRATION

#include <Eigen/Core>
#include <random>
#include <chrono>
#include <iostream>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <queue>

#include <pcl/io/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>


typedef Eigen::Matrix<float, 3, Eigen::Dynamic> cloudPoints;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> cloudValues;

struct _RotaAxis2DNode
{
    float theta_l, theta_r, phi_l, phi_r;
    float ub, lb;
    friend bool operator < (const struct _RotaAxis2DNode & n1, const struct _RotaAxis2DNode & n2)
    {
        if(n1.lb != n2.lb)
            return n1.lb > n2.lb;
        else
            return (n1.theta_r - n1.theta_l)*(n1.phi_r - n1.phi_l) < (n2.theta_r - n2.theta_l)*(n2.phi_r - n2.phi_l);
    }
};
typedef _RotaAxis2DNode RotaAxis2DNode;

struct _RotaAxis1DNode
{
    float theta_l, theta_r;
    float ub, lb;
    friend bool operator < (const struct _RotaAxis1DNode & n1, const struct _RotaAxis1DNode & n2)
    {
        if(n1.lb != n2.lb)
            return n1.lb > n2.lb;
        else
            return (n1.theta_r - n1.theta_l) < (n2.theta_r - n2.theta_l);
    }
};
typedef _RotaAxis1DNode RotaAxis1DNode;


namespace TEAR{

    class registration{

    public:

        registration();
        ~registration();


        float BRANCH_ACCURACY = 5e-2;
        float ErrorThre = 5e-3;

        float noise_bound;


        std::vector<size_t> indices_ori_three;
        std::vector<int> label_three;
        std::vector<size_t> indices_ori_four;
        std::vector<int> label_four;


        int num_corrs_first;
        int num_corrs_second;

        float upper_bound_ini_first;
        float upper_bound_ini_second;


        cloudPoints src_points_afterfirst;
        cloudPoints tgt_points_afterfirst;

        cloudPoints src_points_aftersecond;
        cloudPoints tgt_points_aftersecond;


        // Estimated r and t^T*r
        Eigen::Matrix3f est_rotation;
        Eigen::Vector3f est_translation;

        Eigen::Vector3f rotation_axis_estfirst;
        float tanslation_estfirst;
        Eigen::Vector3f rotation_axis_estsecond;
        float tanslation_estsecond;


        void TEAR_est(cloudPoints &src_points, cloudPoints &tgt_points);

        void TEAR_3DoF(cloudPoints &src_points, cloudPoints &tgt_points, float &xi_first, std::vector<float> &residual_error_first);

        void TEAR_2DoF(cloudPoints &src_points, cloudPoints &tgt_points, std::vector<float> &xi_second, std::vector<float> &residual_error_second);

    protected:

        float GlobalScale;

        void init_indices();

        float upper_bound_3DoF(cloudPoints &src_points, cloudValues &tgt_first, float &xi_first, RotaAxis2DNode &node, float &t_first);

        float lower_bound_3DoF(cloudPoints &src_points, cloudValues &tgt_first, float &xi_first, RotaAxis2DNode &node);

        float upper_bound_2DoF(cloudPoints &src_points, cloudValues &tgt_second, std::vector<float> &xi_second, RotaAxis1DNode &node, float &t_second);

        float lower_bound_2DoF(cloudPoints &src_points, cloudValues &tgt_second, std::vector<float> &xi_second, RotaAxis1DNode &node);

        template <typename T>
        std::vector<size_t> sort_indexes(std::vector<T> &v);

        void bound_multipli_3DoF(Eigen::Vector3f &vec, RotaAxis2DNode &node, float &edge_l, float &edge_r);

        void bound_multipli_2DoF(Eigen::Vector3f &vec, RotaAxis1DNode &node, float &edge_l, float &edge_r);

        void post_refinement(cloudPoints &src_points, cloudPoints &tgt_points, Eigen::Matrix3f &rotation, Eigen::Vector3f &translation, int num);

        void post_SVD(cloudPoints &src_points, cloudPoints &tgt_points, Eigen::Matrix3f &R_pre, Eigen::Vector3f &t_pre, Eigen::Matrix3f &R_post, Eigen::Vector3f &t_post);
    };


}


#endif
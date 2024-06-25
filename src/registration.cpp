#include "registration.h"

#include <random>
#include <stdlib.h>

using namespace TEAR;

registration::registration() = default;

registration::~registration() = default;

void registration::init_indices() {
    for (int i = 0; i < num_corrs_first; i++){
        indices_ori_three.push_back(3*i);
        indices_ori_three.push_back(3*i+1);
        indices_ori_three.push_back(3*i+2);
        label_three.push_back(0);
        label_three.push_back(1);
        label_three.push_back(2);

        indices_ori_four.push_back(4*i);
        indices_ori_four.push_back(4*i+1);
        indices_ori_four.push_back(4*i+2);
        indices_ori_four.push_back(4*i+3);
        label_four.push_back(0);
        label_four.push_back(1);
        label_four.push_back(2);
        label_four.push_back(3);
    }
}


void registration::TEAR_est(cloudPoints &src_points, cloudPoints &tgt_points) {

    cloudPoints src_points_est = src_points;
    cloudPoints tgt_points_est = tgt_points;

    int N_ori = src_points.cols();

    num_corrs_first = src_points_est.cols();
    init_indices();

    // first stage
    float xi_first = noise_bound;
    upper_bound_ini_first = num_corrs_first * xi_first;
    std::vector<float> residual_error_first;
    TEAR_3DoF(src_points_est, tgt_points_est, xi_first, residual_error_first);
    src_points_afterfirst = src_points_est;
    tgt_points_afterfirst = tgt_points_est;

    //second stage
    num_corrs_second = src_points_est.cols();
    std::vector<float> xi_second;
    upper_bound_ini_second = 0;
    for(int i = 0; i < num_corrs_second; i++){
//        float xi_second_i = xi_first - residual_error_first[i];
        float xi_second_i = noise_bound;
        xi_second.push_back(xi_second_i);
        upper_bound_ini_second += xi_second_i;
    }
    std::vector<float> residual_error_second;
    TEAR_2DoF(src_points_est, tgt_points_est, xi_second, residual_error_second);
    src_points_aftersecond = src_points_est;
    tgt_points_aftersecond = tgt_points_est;

    Eigen::Matrix4f transformation_matrix_trtheta;
    transformation_matrix_trtheta = pcl::umeyama(src_points_est, tgt_points_est);
    Eigen::Matrix3f Rotation_ini = transformation_matrix_trtheta.block<3,3>(0, 0);
    Eigen::Vector3f Translation_ini = transformation_matrix_trtheta.block<3,1>(0, 3);

    post_refinement(src_points, tgt_points, Rotation_ini, Translation_ini, 5);

    est_rotation = Rotation_ini;
    est_translation = Translation_ini;
}




void registration::TEAR_3DoF(cloudPoints &src_points, cloudPoints &tgt_points, float &xi_first, std::vector<float> &residual_error_first) {

    // branch and bound search on rotation axis  r = [sin(phi)cos(theta)， sin(phi)sin(theta)，cos(phi)]^T
    //                                             theta \in [0, pi]    phi \in [0, pi]

    cloudValues tgt_first = tgt_points.row(0);

    float est_t_first;
    float optupbound = 1e30;
    RotaAxis2DNode opt_node;


    int bnb_i;
#pragma omp parallel for default(none) shared(src_points, tgt_first, xi_first, optupbound, \
                                              ErrorThre, opt_node, est_t_first) private(bnb_i) num_threads(12)
    for(bnb_i = 0; bnb_i < 12; bnb_i++){
        RotaAxis2DNode node, nodeParent;
        std::priority_queue<RotaAxis2DNode> queueRotaAxis;

        RotaAxis2DNode Initnode;
        Initnode.theta_l = (bnb_i % 4) * M_PI / 4;
        Initnode.theta_r = (bnb_i % 4 + 1) * M_PI / 4;
        Initnode.phi_l = (bnb_i % 3) * M_PI / 3;
        Initnode.phi_r = (bnb_i % 3 + 1) * M_PI / 3;

//        std::vector<size_t> indices_init;
        float t_first_init;
        Initnode.lb = lower_bound_3DoF(src_points, tgt_first, xi_first, Initnode);
        Initnode.ub = upper_bound_3DoF(src_points, tgt_first, xi_first, Initnode, t_first_init);
        queueRotaAxis.push(Initnode);

        if (Initnode.ub < optupbound) {
#pragma omp critical
            {
                optupbound = Initnode.ub;
                est_t_first = t_first_init;
                opt_node = Initnode;
            }
        }

        while(1){

            if(queueRotaAxis.empty())
                break;

            nodeParent = queueRotaAxis.top();
            queueRotaAxis.pop();

            if(optupbound - nodeParent.lb <= ErrorThre)
                break;

            float theta_width = nodeParent.theta_r - nodeParent.theta_l;
            float phi_width = nodeParent.phi_r - nodeParent.phi_l;

            if(theta_width <= BRANCH_ACCURACY || phi_width <= BRANCH_ACCURACY)
                break;

            for(int i = 0; i < 2; i++){

                node.theta_l = nodeParent.theta_l + i*0.5*theta_width;
                node.theta_r = nodeParent.theta_l + (0.5+i*0.5)*theta_width;

                for(int j = 0; j < 2; j++) {
                    node.phi_l = nodeParent.phi_l + j * 0.5 * phi_width;
                    node.phi_r = nodeParent.phi_l + (0.5 + j * 0.5) * phi_width;

                    // calculate the lower bound
                    float node_lb = lower_bound_3DoF(src_points, tgt_first, xi_first, node);
                    if (node_lb >= optupbound) {
                        continue;
                    }
                    node.lb = node_lb;

//                    std::vector<size_t> indices_node;
                    float t_first_node;

                    float node_ub = upper_bound_3DoF(src_points, tgt_first, xi_first, node, t_first_node);
                    node.ub = node_ub;
                    if (node_ub < optupbound) {
#pragma omp critical
                        {
                            optupbound = node_ub;
                            est_t_first = t_first_node;
                            opt_node = node;
                        }
                    }
                    queueRotaAxis.push(node);
                }
            }
        }
    }

    float theta_est = (opt_node.theta_l + opt_node.theta_r) / 2;
    float phi_est = (opt_node.phi_l + opt_node.phi_r) / 2;

    rotation_axis_estfirst << std::sin(phi_est)*std::cos(theta_est),
            std::sin(phi_est)*std::sin(theta_est),
            std::cos(phi_est);

    tanslation_estfirst = est_t_first;

    std::vector<int> indices_est;
    for(int m = 0; m < num_corrs_first;m++){

        float error_m = xi_first - std::abs(tgt_first.col(m).value() - rotation_axis_estfirst.dot(src_points.col(m)) - tanslation_estfirst);

        if(error_m > 0){
            indices_est.push_back(m);
        }
    }

    int num_est = indices_est.size();
    cloudPoints src_est_first(3, num_est);
    cloudPoints tgt_est_first(3, num_est);

    for(int n = 0; n< num_est; n++){
        int index = indices_est[n];
        src_est_first.col(n) = src_points.col(index);
        tgt_est_first.col(n) = tgt_points.col(index);
    }

    src_points = src_est_first;
    tgt_points = tgt_est_first;
}


float registration::upper_bound_3DoF(cloudPoints &src_points, cloudValues &tgt_first, float &xi_first, RotaAxis2DNode &node, float &t_first) {


    float theta_center = (node.theta_l + node.theta_r) / 2;
    float phi_center = (node.phi_l + node.phi_r) / 2;

    Eigen::Vector3f rotation_axis_center;
    rotation_axis_center[0] = std::sin(phi_center)*std::cos(theta_center);
    rotation_axis_center[1] = std::sin(phi_center)*std::sin(theta_center);
    rotation_axis_center[2] = std::cos(phi_center);


    // sorted indices
    std::vector<float> lamda_list;
    std::vector<size_t> indices(indices_ori_three);
    for (int i = 0; i < num_corrs_first; i++){
        float a_i = tgt_first.col(i).value() - src_points.col(i).dot(rotation_axis_center);
        lamda_list.push_back(a_i - xi_first);
        lamda_list.push_back(a_i);
        lamda_list.push_back(a_i + xi_first);
    }

    sort(indices.begin(), indices.end(),
         [&lamda_list](size_t i1, size_t i2) { return lamda_list[i1] < lamda_list[i2]; });

    std::vector<float> sorted_lamda;
    for (int j = 0; j < 3*num_corrs_first; j++){
        sorted_lamda.push_back(lamda_list[indices[j]]);
    }

    // initialization
    float upper_bound_m = num_corrs_first * xi_first;
    float upper_bound_opt = upper_bound_m;
    int up_num = 0;
    int down_num = 0;

    for(int m = 0; m < 3*num_corrs_first-1; m++){

        float lamda_k = sorted_lamda[m];
        int index = indices[m];
        int index_type = label_three[index];

        if(index_type == 1){
            up_num++;
            down_num--;

            if(upper_bound_m < upper_bound_opt){
                t_first = lamda_k;
                upper_bound_opt = upper_bound_m;
            }
        }
        else if (index_type == 0){
            down_num++;
        }
        else{
            up_num--;
        }

        upper_bound_m += (up_num - down_num) * (sorted_lamda[m+1] - lamda_k);
    }

    return upper_bound_opt;
}



float registration::lower_bound_3DoF(cloudPoints &src_points, cloudValues &tgt_first, float &xi_first, RotaAxis2DNode &node) {

    std::vector<float> lamda_list;
    std::vector<size_t> indices(indices_ori_four);

    for (int i = 0; i < num_corrs_first; i++){
        float edge_l;
        float edge_r;
        Eigen::Vector3f src_vec_i = src_points.col(i);
        bound_multipli_3DoF(src_vec_i, node, edge_l, edge_r);

        float b_l = tgt_first.col(i).value() - edge_r;
        float b_r = tgt_first.col(i).value() - edge_l;

        lamda_list.push_back(b_l - xi_first);
        lamda_list.push_back(b_l);
        lamda_list.push_back(b_r);
        lamda_list.push_back(b_r + xi_first);
    }


    sort(indices.begin(), indices.end(),
         [&lamda_list](size_t i1, size_t i2) { return lamda_list[i1] < lamda_list[i2]; });

    std::vector<float> sorted_lamda;
    for (int j = 0; j < 4*num_corrs_first; j++){
        sorted_lamda.push_back(lamda_list[indices[j]]);
    }

    // initialization
    float lower_bound_m = upper_bound_ini_first;
    float lower_bound_opt = lower_bound_m;
    int up_num = 0;
    int down_num = 0;

    for(int m = 0; m < 4*num_corrs_first-1; m++){

        float lamda_k = sorted_lamda[m];
        int index = indices[m];
        int index_type = label_four[index];

        if(index_type == 1){
            down_num--;

            if(lower_bound_m < lower_bound_opt){
                lower_bound_opt = lower_bound_m;
            }
        }
        else if(index_type == 2){
            up_num++;

            if(lower_bound_m < lower_bound_opt){
                lower_bound_opt = lower_bound_m;
            }
        }
        else if (index_type == 0){
            down_num++;
        }
        else{
            up_num--;
        }

        lower_bound_m += (up_num - down_num) * (sorted_lamda[m+1] - lamda_k);
    }

    return lower_bound_opt;
}


void registration::bound_multipli_3DoF(Eigen::Vector3f &vec, RotaAxis2DNode &node, float &edge_l, float &edge_r) {

    float theta_l = node.theta_l;
    float theta_r = node.theta_r;
    float phi_l = node.phi_l;
    float phi_r = node.phi_r;

    float inner_lb;
    float inner_ub;

    //  get the f_1 and f_2
    float inner_norm = sqrt(vec[0]*vec[0] + vec[1]*vec[1]);
    if(vec[1] < 0){                   // sin < 0 -> more than pi
        inner_norm = -inner_norm;
    }

    float inner_lamda_1 = vec[0] / inner_norm;

    float inner_candi_theta_l = vec[0] * std::cos(theta_l) + vec[1]* std::sin(theta_l);
    float inner_candi_theta_r = vec[0] * std::cos(theta_r) + vec[1]* std::sin(theta_r);
    if (inner_lamda_1 <= std::cos(theta_r) || inner_lamda_1 >= std::cos(theta_l)){
        if (inner_candi_theta_l < inner_candi_theta_r){
            inner_lb = inner_candi_theta_l;
            inner_ub = inner_candi_theta_r;
        }else{
            inner_lb = inner_candi_theta_r;
            inner_ub = inner_candi_theta_l;
        }
    }
    else{
        if(inner_norm >= 0){
            float lower = std::min(inner_candi_theta_l, inner_candi_theta_r);
            inner_lb = lower;
            inner_ub = inner_norm;
        }
        else{
            float upper= std::max(inner_candi_theta_l, inner_candi_theta_r);
            inner_lb = inner_norm;
            inner_ub = upper;
        }
    }

    // get the lower bound of vec[2]*cos(phi) + inner_lb*sin(phi)
    float outer_norm_lb = sqrt(vec[2]*vec[2] + inner_lb*inner_lb);
    if(inner_lb < 0){
        outer_norm_lb = -outer_norm_lb;
    }
    float outer_lamba_1_lb = vec[2] / outer_norm_lb;

    float outer_candi_phi_l_lb = vec[2] * std::cos(phi_l) + inner_lb*std::sin(phi_l);
    float outer_candi_phi_r_lb = vec[2] * std::cos(phi_r) + inner_lb*std::sin(phi_r);
    if(outer_lamba_1_lb <= std::cos(phi_r) || outer_lamba_1_lb >= std::cos(phi_l)){
        edge_l = std::min(outer_candi_phi_l_lb, outer_candi_phi_r_lb);
    }
    else{
        if(outer_norm_lb >= 0){
            float lower = std::min(outer_candi_phi_l_lb, outer_candi_phi_l_lb);
            edge_l = lower;
        }
        else{
            edge_l = outer_norm_lb;
        }
    }

    // get the upper bound of vec[2]*cos(phi) + inner_ub*sin(phi)
    float outer_norm_ub = sqrt(vec[2]*vec[2] + inner_ub*inner_ub);
    if(inner_ub < 0){
        outer_norm_ub = -outer_norm_ub;
    }
    float outer_lamba_1_ub = vec[2] / outer_norm_ub;

    float outer_candi_phi_l_ub = vec[2] * std::cos(phi_l) + inner_ub*std::sin(phi_l);
    float outer_candi_phi_r_ub = vec[2] * std::cos(phi_r) + inner_ub*std::sin(phi_r);
    if(outer_lamba_1_ub <= std::cos(phi_r) || outer_lamba_1_ub >= std::cos(phi_l)){
        edge_r = std::max(outer_candi_phi_l_ub, outer_candi_phi_r_ub);
    }
    else{
        if(outer_norm_ub > 0){
            edge_r = outer_norm_ub;
        }
        else{
            float upper = std::max(outer_candi_phi_l_ub, outer_candi_phi_r_ub);
            edge_r = upper;
        }
    }

    assert(edge_l < edge_r);

}

void registration::TEAR_2DoF(cloudPoints &src_points, cloudPoints &tgt_points, std::vector<float> &xi_second,
                             std::vector<float> &residual_error_second) {

    cloudValues tgt_second = tgt_points.row(1);

    float est_t_second;
    float optupbound = 1e9;
    RotaAxis1DNode opt_node;

    float theta_0 = std::atan(-rotation_axis_estfirst(0) / rotation_axis_estfirst(1));
    if(theta_0<0){
        theta_0 += M_PI;
    }
    float theta_1 = theta_0 + M_PI;

    std::vector<float> endpoints;
    endpoints.push_back(0);
    for(int i = 0; i < 10; i++){
        float end_j = (i+1) * M_PI / 5;
        if(*(endpoints.end()-1) < theta_0 && theta_0 < end_j){
            endpoints.push_back(theta_0-1e-4);
            endpoints.push_back(theta_0+1e-4);
        }
        else if(*(endpoints.end()-1) < theta_1 && theta_1 < end_j){
            endpoints.push_back(theta_1-1e-4);
            endpoints.push_back(theta_1+1e-4);
        }
        endpoints.push_back(end_j);
        if(i < 9){
            endpoints.push_back(end_j);
        }
    }


    int bnb_i;
#pragma omp parallel for default(none) shared(src_points, tgt_second, xi_second, endpoints, optupbound, \
                                              ErrorThre, opt_node, est_t_second) private(bnb_i) num_threads(12)
    for(bnb_i = 0; bnb_i < 12; bnb_i++){

        RotaAxis1DNode  node, nodeParent;
        std::priority_queue<RotaAxis1DNode> queueRotaAxis;

        RotaAxis1DNode Initnode;
        Initnode.theta_l = endpoints[bnb_i];
        Initnode.theta_l = endpoints[bnb_i + 1];

        float t_second_init;
        Initnode.lb = lower_bound_2DoF(src_points, tgt_second, xi_second, Initnode);
        Initnode.ub = upper_bound_2DoF(src_points, tgt_second, xi_second, Initnode, t_second_init);

        if (Initnode.ub < optupbound) {
#pragma omp critical
            {
                optupbound = Initnode.ub;
                est_t_second = t_second_init;
                opt_node = Initnode;
            }
        }

        while(1){

            if(queueRotaAxis.empty())
                break;

            nodeParent = queueRotaAxis.top();
            queueRotaAxis.pop();

            if(optupbound - nodeParent.lb <= ErrorThre)
                break;

            float theta_width = nodeParent.theta_r - nodeParent.theta_l;

            if(theta_width <= BRANCH_ACCURACY)
                break;

            for(int i = 0; i < 2; i++){

                node.theta_l = nodeParent.theta_l + i*0.5*theta_width;
                node.theta_r = nodeParent.theta_l + (0.5+i*0.5)*theta_width;

                // calculate the lower bound
                float node_lb = lower_bound_2DoF(src_points, tgt_second, xi_second, node);
                if (node_lb >= optupbound) {
                    continue;
                }
                node.lb = node_lb;

                float t_second_node;

                float node_ub = upper_bound_2DoF(src_points, tgt_second, xi_second, node, t_second_node);
                node.ub = node_ub;
                if(node_ub < optupbound){
#pragma omp critical
                    {
                        optupbound = node_ub;
                        est_t_second = t_second_node;
                        opt_node = node;
                    }
                }
                queueRotaAxis.push(node);
            }
        }
    }

    float theta_est = (opt_node.theta_l + opt_node.theta_r) / 2;
    float tan_phi = - rotation_axis_estfirst(2) / (rotation_axis_estfirst(0)*std::cos(theta_est) + rotation_axis_estfirst(1)*std::cos(theta_est));
    float phi_est = std::atan(tan_phi);
    if(phi_est < 0){
        phi_est += M_PI;
    }

    rotation_axis_estsecond << std::sin(phi_est)*std::cos(theta_est),
            std::sin(phi_est)*std::sin(theta_est),
            std::cos(phi_est);

    tanslation_estsecond = est_t_second;

    std::vector<int> indices_est;
    for(int m = 0; m < num_corrs_second;m++){

        float error_m = xi_second[m] - std::abs(tgt_second.col(m).value() - rotation_axis_estsecond.dot(src_points.col(m)) - tanslation_estsecond);

        if(error_m > 0){
            indices_est.push_back(m);
        }
    }

    int num_est = indices_est.size();
    cloudPoints src_est_second(3, num_est);
    cloudPoints tgt_est_second(3, num_est);

    for(int n = 0; n< num_est; n++){
        int index = indices_est[n];
        src_est_second.col(n) = src_points.col(index);
        tgt_est_second.col(n) = tgt_points.col(index);
    }

    src_points = src_est_second;
    tgt_points = tgt_est_second;
}

float registration::upper_bound_2DoF(cloudPoints &src_points, cloudValues &tgt_second, std::vector<float> &xi_second,
                                     RotaAxis1DNode &node, float &t_second) {

    float theta_est = (node.theta_l + node.theta_r) / 2;
    float tan_phi = - rotation_axis_estfirst(2) / (rotation_axis_estfirst(0)*std::cos(theta_est) + rotation_axis_estfirst(1)*std::cos(theta_est));
    float phi_est = std::atan(tan_phi);
    if(phi_est < 0){
        phi_est += M_PI;
    }

    Eigen::Vector3f rotation_axis_center;
    rotation_axis_center << std::sin(phi_est)*std::cos(theta_est),
            std::sin(phi_est)*std::sin(theta_est),
            std::cos(phi_est);

    // sorted indices
    std::vector<float> lamda_list;
    std::vector<size_t> indices(indices_ori_three.begin(), indices_ori_three.begin() + 3*num_corrs_second);
    for (int i = 0; i < num_corrs_second; i++){
        float a_i = tgt_second.col(i).value() - src_points.col(i).dot(rotation_axis_center);
        lamda_list.push_back(a_i - xi_second[i]);
        lamda_list.push_back(a_i);
        lamda_list.push_back(a_i + xi_second[i]);
    }

    sort(indices.begin(), indices.end(),
         [&lamda_list](size_t i1, size_t i2) { return lamda_list[i1] < lamda_list[i2]; });

    std::vector<float> sorted_lamda;
    for (int j = 0; j < 3*num_corrs_second; j++){
        sorted_lamda.push_back(lamda_list[indices[j]]);
    }

    // initialization
    float upper_bound_m = upper_bound_ini_second;
    float upper_bound_opt = upper_bound_m;
    int up_num = 0;
    int down_num = 0;

    for(int m = 0; m < 3*num_corrs_second - 1; m++){

        float lamda_k = sorted_lamda[m];
        int index = indices[m];
        int index_type = label_three[index];

        if(index_type == 1){
            up_num++;
            down_num--;

            if(upper_bound_m < upper_bound_opt){
                t_second = lamda_k;
                upper_bound_opt = upper_bound_m;
            }
        }
        else if (index_type == 0){
            down_num++;
        }
        else{
            up_num--;
        }

        upper_bound_m += (up_num - down_num) * (sorted_lamda[m+1] - lamda_k);

    }

    return upper_bound_opt;
}

float registration::lower_bound_2DoF(cloudPoints &src_points, cloudValues &tgt_second, std::vector<float> &xi_second,
                                     RotaAxis1DNode &node) {

    std::vector<float> lamda_list;
    std::vector<size_t> indices(indices_ori_four.begin(), indices_ori_four.begin() + 4*num_corrs_second);

    for(int i = 0; i < num_corrs_second; i++){
        float edge_l;
        float edge_r;
        Eigen::Vector3f src_vec_i = src_points.col(i);
        bound_multipli_2DoF(src_vec_i, node, edge_l, edge_r);

        float b_l = tgt_second.col(i).value() - edge_r;
        float b_r = tgt_second.col(i).value() - edge_l;

        lamda_list.push_back(b_l - xi_second[i]);
        lamda_list.push_back(b_l);
        lamda_list.push_back(b_r);
        lamda_list.push_back(b_r + xi_second[i]);
    }

    sort(indices.begin(), indices.end(),
         [&lamda_list](size_t i1, size_t i2) { return lamda_list[i1] < lamda_list[i2]; });

    std::vector<float> sorted_lamda;
    for (int j = 0; j < 4*num_corrs_second; j++){
        sorted_lamda.push_back(lamda_list[indices[j]]);
    }

    // initialization
    float lower_bound_m = upper_bound_ini_second;
    float lower_bound_opt = lower_bound_m;
    int up_num = 0;
    int down_num = 0;

    for(int m = 0; m < 4*num_corrs_second-1; m++){

        float lamda_k = sorted_lamda[m];
        int index = indices[m];
        int index_type = label_four[index];

        if(index_type == 1){
            down_num--;

            if(lower_bound_m < lower_bound_opt){
                lower_bound_opt = lower_bound_m;
            }
        }
        else if(index_type == 2){
            up_num++;

            if(lower_bound_m < lower_bound_opt){
                lower_bound_opt = lower_bound_m;
            }
        }
        else if (index_type == 0){
            down_num++;
        }
        else{
            up_num--;
        }

        lower_bound_m += (up_num - down_num) * (sorted_lamda[m+1] - lamda_k);
    }

    return lower_bound_opt;

}

void registration::bound_multipli_2DoF(Eigen::Vector3f &vec, RotaAxis1DNode &node, float &edge_l, float &edge_r) {


    float theta_l = node.theta_l;
    float theta_r = node.theta_r;

    float inner_lb;
    float inner_ub;

    //  get the f_1 and f_2
    float inner_norm = sqrt(vec[0]*vec[0] + vec[1]*vec[1]);
    if(vec[1] < 0){                   // sin < 0 -> more than pi
        inner_norm = -inner_norm;
    }

    float inner_lamda_1 = vec[0] / inner_norm;

    float inner_candi_theta_l = vec[0] * std::cos(theta_l) + vec[1]* std::sin(theta_l);
    float inner_candi_theta_r = vec[0] * std::cos(theta_r) + vec[1]* std::sin(theta_r);

    if(theta_l < M_PI){
        if (inner_lamda_1 <= std::cos(theta_r) || inner_lamda_1 >= std::cos(theta_l)){
            if (inner_candi_theta_l < inner_candi_theta_r){
                inner_lb = inner_candi_theta_l;
                inner_ub = inner_candi_theta_r;
            }else{
                inner_lb = inner_candi_theta_r;
                inner_ub = inner_candi_theta_l;
            }
        }
        else{
            if(inner_norm >= 0){
                float lower = std::min(inner_candi_theta_l, inner_candi_theta_r);
                inner_lb = lower;
                inner_ub = inner_norm;
            }
            else{
                float upper = std::max(inner_candi_theta_l, inner_candi_theta_r);
                inner_lb = inner_norm;
                inner_ub = upper;
            }
        }
    }
    else{
        if(inner_lamda_1 >= std::cos(theta_l-M_PI) || inner_lamda_1 <= std::cos(theta_r-M_PI)){
            if (inner_candi_theta_l < inner_candi_theta_r){
                inner_lb = inner_candi_theta_l;
                inner_ub = inner_candi_theta_r;
            }else{
                inner_lb = inner_candi_theta_r;
                inner_ub = inner_candi_theta_l;
            }
        }
        else{
            if(inner_norm >= 0){
                float upper = std::max(inner_candi_theta_l, inner_candi_theta_r);
                inner_lb = inner_norm;
                inner_ub = upper;
            }
            else{
                float lower = std::min(inner_candi_theta_l, inner_candi_theta_r);
                inner_lb = lower;
                inner_ub = inner_norm;
            }
        }
    }

    // get the range of phi
    float tan_phi_1 = - rotation_axis_estfirst(2) / (rotation_axis_estfirst(0)*std::cos(theta_l) + rotation_axis_estfirst(1)*std::cos(theta_l));
    float phi_1 = std::atan(tan_phi_1);
    float tan_phi_2 = - rotation_axis_estfirst(2) / (rotation_axis_estfirst(0)*std::cos(theta_r) + rotation_axis_estfirst(1)*std::cos(theta_r));
    float phi_2 = std::atan(tan_phi_2);

    assert(phi_1 * phi_2 > 0);
    float phi_l, phi_r;
    if(phi_1 > 0){
        if(phi_1 < phi_2){
            phi_l = phi_1;
            phi_r = phi_2;
        }else{
            phi_l = phi_2;
            phi_r = phi_1;
        }
    }
    else{
        if(phi_1 < phi_2){
            phi_l = -phi_2;
            phi_r = -phi_1;
        }else{
            phi_l = -phi_1;
            phi_r = -phi_2;
        }
    }

    // get the lower bound of vec[2]*cos(phi) + inner_lb*sin(phi)
    float outer_norm_lb = sqrt(vec[2]*vec[2] + inner_lb*inner_lb);
    if(inner_lb < 0){
        outer_norm_lb = -outer_norm_lb;
    }
    float outer_lamba_1_lb = vec[2] / outer_norm_lb;

    float outer_candi_phi_l_lb = vec[2] * std::cos(phi_l) + inner_lb*std::sin(phi_l);
    float outer_candi_phi_r_lb = vec[2] * std::cos(phi_r) + inner_lb*std::sin(phi_r);
    if(outer_lamba_1_lb <= std::cos(phi_r) || outer_lamba_1_lb >= std::cos(phi_l)){
        edge_l = std::min(outer_candi_phi_l_lb, outer_candi_phi_r_lb);
    }
    else{
        if(outer_norm_lb >= 0){
            float lower = std::min(outer_candi_phi_l_lb, outer_candi_phi_l_lb);
            edge_l = lower;
        }
        else{
            edge_l = outer_norm_lb;
        }
    }

    // get the upper bound of vec[2]*cos(phi) + inner_ub*sin(phi)
    float outer_norm_ub = sqrt(vec[2]*vec[2] + inner_ub*inner_ub);
    if(inner_ub < 0){
        outer_norm_ub = -outer_norm_ub;
    }
    float outer_lamba_1_ub = vec[2] / outer_norm_ub;

    float outer_candi_phi_l_ub = vec[2] * std::cos(phi_l) + inner_ub*std::sin(phi_l);
    float outer_candi_phi_r_ub = vec[2] * std::cos(phi_r) + inner_ub*std::sin(phi_r);
    if(outer_lamba_1_ub <= std::cos(phi_r) || outer_lamba_1_ub >= std::cos(phi_l)){
        edge_r = std::max(outer_candi_phi_l_ub, outer_candi_phi_r_ub);
    }
    else{
        if(outer_norm_ub > 0){
            edge_r = outer_norm_ub;
        }
        else{
            float upper = std::max(outer_candi_phi_l_ub, outer_candi_phi_r_ub);
            edge_r = upper;
        }
    }

    assert(edge_l < edge_r);

}


template <typename T>
std::vector<size_t> registration::sort_indexes(std::vector<T> &v)
{
    std::vector<size_t> idx(v.size());

    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

    return idx;
}

void registration::post_refinement(cloudPoints &src_points, cloudPoints &tgt_points, Eigen::Matrix3f &rotation,
                                   Eigen::Vector3f &translation, int num) {

    Eigen::Matrix3f R_flag;
    Eigen::Vector3f t_flag;

    R_flag = rotation;
    t_flag = translation;
    for(int i = 0; i< num; i++){

        post_SVD(src_points, tgt_points, R_flag, t_flag, rotation, translation);

        R_flag = rotation;
        t_flag = translation;
    }

}

void registration::post_SVD(cloudPoints &src_points, cloudPoints &tgt_points, Eigen::Matrix3f &R_pre,
                            Eigen::Vector3f &t_pre, Eigen::Matrix3f &R_post, Eigen::Vector3f &t_post) {

    int N_cor = src_points.cols();
    int inliers_num = 0;
    std::vector<int> inliers_index;
    
    for(int i =0 ;i<N_cor;i++){

        Eigen::Vector3f src_i_vector = src_points.col(i);
        Eigen::Vector3f tgt_i_vector = tgt_points.col(i);

        float error = (tgt_i_vector - R_pre * src_i_vector - t_pre).norm();

        if(error <= 2*noise_bound){
            inliers_index.push_back(i);
            inliers_num++;
        }
    }

    cloudPoints inliers_src(3, inliers_num);
    cloudPoints inliers_tgt(3, inliers_num);

    for(int j = 0; j <inliers_num ;j++){
        inliers_src.col(j) = src_points.col(inliers_index[j]);
        inliers_tgt.col(j) = tgt_points.col(inliers_index[j]);
    }

    Eigen::Matrix4f transformation_matrix;
    transformation_matrix = pcl::umeyama(inliers_src, inliers_tgt);
    R_post = transformation_matrix.block<3,3>(0, 0);
    t_post = transformation_matrix.block<3,1>(0, 3);

}
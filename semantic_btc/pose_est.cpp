#include "include/btc.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <map>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;

// Pose structure with full transformation (translation + rotation)
struct Pose {
    Eigen::Vector3d translation;
    Eigen::Matrix3d rotation;
    bool valid{false};
    
    Pose() : translation(Eigen::Vector3d::Zero()), rotation(Eigen::Matrix3d::Identity()), valid(false) {}
};

// 从文件名中提取序号
int extractNumberFromFilename(const std::string& filename) {
    std::string base = fs::path(filename).stem().string();
    return std::stoi(base);
}

// Read KITTI format pose file (3x4 matrix per line, 12 numbers space-separated)
static std::vector<Pose> readKittiPoses(const std::string& pose_file) {
    std::vector<Pose> poses;
    std::ifstream in(pose_file);
    if (!in.is_open()) {
        std::cerr << "Could not open pose file: " << pose_file << std::endl;
        return poses;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        double m[12];
        bool ok = true;
        for (int i = 0; i < 12; ++i) {
            if (!(iss >> m[i])) { ok = false; break; }
        }
        Pose p;
        if (ok) {
            // KITTI format: 3x4 matrix
            // [R00 R01 R02 tx]
            // [R10 R11 R12 ty]
            // [R20 R21 R22 tz]
            p.rotation << m[0], m[1], m[2],
                          m[4], m[5], m[6],
                          m[8], m[9], m[10];
            p.translation << m[3], m[7], m[11];
            p.valid = true;
        }
        poses.push_back(p);
    }
    return poses;
}

// Calculate distance between two poses (translation only)
static inline double distance3(const Pose& a, const Pose& b) {
    if (!a.valid || !b.valid) return -1.0;
    Eigen::Vector3d diff = a.translation - b.translation;
    return diff.norm();
}

// Convert pose (translation + rotation) to 4x4 transformation matrix
static Eigen::Matrix4d pose_to_matrix(const Pose& pose) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = pose.rotation;  // Rotation
    T.block<3, 1>(0, 3) = pose.translation;  // Translation
    return T;
}

// Compute relative pose: T_rel = T2^(-1) * T1 (transformation from frame1 to frame2)
static Eigen::Matrix4d compute_relative_pose(
    const Eigen::Matrix4d& T1, const Eigen::Matrix4d& T2) {
    return T2.inverse() * T1;
}

// Compute pose error between estimated and ground truth relative poses
static void compute_pose_error(
    const Eigen::Matrix4d& T_estimated, const Eigen::Matrix4d& T_ground_truth,
    double& translation_error, double& rotation_error) {
    // Translation error (Euclidean distance)
    Eigen::Vector3d t_est = T_estimated.block<3, 1>(0, 3);
    Eigen::Vector3d t_gt = T_ground_truth.block<3, 1>(0, 3);
    translation_error = (t_est - t_gt).norm();
    
    // Rotation error (angle in degrees)
    Eigen::Matrix3d R_est = T_estimated.block<3, 3>(0, 0);
    Eigen::Matrix3d R_gt = T_ground_truth.block<3, 3>(0, 0);
    Eigen::Matrix3d R_error = R_est * R_gt.transpose();
    
    // Extract angle from rotation matrix
    Eigen::AngleAxisd angle_axis(R_error);
    rotation_error = std::abs(angle_axis.angle()) * 180.0 / M_PI;
}

// 计算准确率和召回率
std::pair<double, double> calculateMetrics(
    const std::vector<std::pair<double, bool>>& results) {
    
    int true_positives = 0;
    int false_positives = 0;
    int false_negatives = 0;
    
    for (const auto& result : results) {
        if (result.first) { // 如果预测为匹配
            if (result.second) { // 如果是真实匹配
                true_positives++;
            } else {
                false_positives++;
            }
        } else if (!result.first) { // 如果预测为不匹配
            if (result.second) {
                false_negatives++;
            }
        }
    }
    
    double precision = (true_positives + false_positives == 0) ? 0.0 :
        static_cast<double>(true_positives) / (true_positives + false_positives);
    double recall = (true_positives + false_negatives == 0) ? 0.0 :
        static_cast<double>(true_positives) / (true_positives + false_negatives);
    
    return {precision, recall};
}

// 获取文件夹中的所有.bin文件
std::vector<std::string> getBinFiles(const std::string& folder) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".bin") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// 获取文件夹中的所有.label文件
std::vector<std::string> getLabelFiles(const std::string& folder) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".label") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// 获取所有子文件夹
std::vector<std::string> getSubfolders(const std::string& folder) {
    std::vector<std::string> subfolders;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.is_directory()) {
            subfolders.push_back(entry.path().string());
        }
    }
    std::sort(subfolders.begin(), subfolders.end());
    return subfolders;
}

// 保存PR曲线数据到CSV文件
void savePRCurveData(const std::string& output_file, 
                    const std::vector<std::tuple<double, bool, double>>& results) {
    std::ofstream out_file(output_file);
    out_file << "Threshold,Precision,Recall\n";

    // 使用所有得分作为阈值点
    std::vector<double> thresholds;
    for (const auto& result : results) {
        thresholds.push_back(std::get<0>(result));
    }
    std::sort(thresholds.begin(), thresholds.end());
    thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());

    // 计算不同阈值下的准确率和召回率
    for (double threshold : thresholds) {
        std::vector<std::pair<double, bool>> threshold_results;
        for (const auto& result : results) {
            // 根据阈值判断预测结果：如果匹配得分 >= 阈值，则预测为匹配(true)，否则预测为不匹配(false)
            bool predicted_match = (std::get<0>(result) >= threshold);
            bool actual_match = std::get<1>(result);
            threshold_results.push_back({predicted_match, actual_match});
        }
        
        auto [precision, recall] = calculateMetrics(threshold_results);
        out_file << threshold << "," << precision << "," << recall << "\n";
    }

    out_file.close();
}

// 详细匹配结果结构（包含ID、距离和位姿）
struct DetailedMatchResult {
    int input_id;
    int target_id;
    double match_score;
    bool is_true_match;
    double computation_time;
    double pose_distance;
    bool has_transform;
    Eigen::Vector3d translation;
    Eigen::Matrix3d rotation;
};

// 保存详细匹配结果到CSV文件（包含匹配时间、ID和距离）
void saveDetailedResults(const std::string& output_file, 
                        const std::vector<DetailedMatchResult>& results) {
    std::ofstream out_file(output_file);
    out_file << "InputID,TargetID,MatchScore,IsTrueMatch,ComputationTimeMs,PoseDistance,HasTransform\n";

    for (const auto& result : results) {
        out_file << result.input_id << ","
                 << result.target_id << ","
                 << result.match_score << ","
                 << (result.is_true_match ? "1" : "0") << ","
                 << result.computation_time << ","
                 << result.pose_distance << ","
                 << (result.has_transform ? "1" : "0") << "\n";
    }

    out_file.close();
}

// 计算Precision=1时的Recall
double calculateRecallAtPrecision1(
    const std::vector<std::tuple<double, bool, double>>& results) {
    
    // 找到所有真正例（true matches）的得分
    std::vector<double> true_match_scores;
    for (const auto& result : results) {
        if (std::get<1>(result)) {  // 如果是真实匹配
            true_match_scores.push_back(std::get<0>(result));
        }
    }
    
    if (true_match_scores.empty()) {
        return 0.0;
    }
    
    // 对真正例得分排序（降序）
    std::sort(true_match_scores.begin(), true_match_scores.end(), std::greater<double>());
    
    // 找到能够达到Precision=1的最大阈值
    // 即：只选择得分最高的真正例，确保没有假正例
    double best_threshold = true_match_scores[0];
    double best_recall = 0.0;
    
    // 尝试不同的阈值，找到Precision=1时最大的Recall
    for (size_t i = 0; i < true_match_scores.size(); ++i) {
        double threshold = true_match_scores[i];
        
        // 计算在这个阈值下的Precision和Recall
        int true_positives = 0;
        int false_positives = 0;
        int total_true_matches = true_match_scores.size();
        
        for (const auto& result : results) {
            double score = std::get<0>(result);
            bool is_true = std::get<1>(result);
            
            if (score >= threshold) {
                if (is_true) {
                    true_positives++;
                } else {
                    false_positives++;
                }
            }
        }
        
        double precision = (true_positives + false_positives == 0) ? 0.0 :
            static_cast<double>(true_positives) / (true_positives + false_positives);
        double recall = (total_true_matches == 0) ? 0.0 :
            static_cast<double>(true_positives) / total_true_matches;
        
        // 如果Precision=1，更新最佳Recall
        if (std::abs(precision - 1.0) < 1e-6) {
            best_recall = std::max(best_recall, recall);
        }
    }
    
    return best_recall;
}

// 计算SR (Success Rate) - 在最佳阈值下的成功率
// 最佳阈值定义为F1-score最大的阈值
double calculateSuccessRate(
    const std::vector<std::tuple<double, bool, double>>& results) {
    
    // 使用所有得分作为阈值点
    std::vector<double> thresholds;
    for (const auto& result : results) {
        thresholds.push_back(std::get<0>(result));
    }
    std::sort(thresholds.begin(), thresholds.end());
    thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
    
    double best_f1 = 0.0;
    double best_threshold = 0.0;
    double best_sr = 0.0;
    
    // 找到F1-score最大的阈值
    for (double threshold : thresholds) {
        std::vector<std::pair<double, bool>> threshold_results;
        for (const auto& result : results) {
            bool predicted_match = (std::get<0>(result) >= threshold);
            bool actual_match = std::get<1>(result);
            threshold_results.push_back({predicted_match, actual_match});
        }
        
        auto [precision, recall] = calculateMetrics(threshold_results);
        
        // 计算F1-score
        double f1 = (precision + recall == 0.0) ? 0.0 :
            2.0 * precision * recall / (precision + recall);
        
        if (f1 > best_f1) {
            best_f1 = f1;
            best_threshold = threshold;
            // SR = 正确预测的比例 = (TP + TN) / Total
            int correct = 0;
            for (const auto& res : threshold_results) {
                if (res.first == res.second) {  // 预测正确
                    correct++;
                }
            }
            best_sr = static_cast<double>(correct) / threshold_results.size();
        }
    }
    
    return best_sr;
}

// 计算平均匹配耗时
double calculateAverageComputationTime(
    const std::vector<std::tuple<double, bool, double>>& results) {
    
    if (results.empty()) {
        return 0.0;
    }
    
    double total_time = 0.0;
    for (const auto& result : results) {
        total_time += std::get<2>(result);  // 第三个元素是计算时间
    }
    
    return total_time / results.size();
}

// 计算平均每帧的描述子提取时间
double calculateAverageDescriptorExtractionTime(
    const std::vector<std::tuple<double, bool, double>>& results) {
    
    if (results.empty()) {
        return 0.0;
    }
    
    double total_time = 0.0;
    for (const auto& result : results) {
        total_time += std::get<2>(result);  // 第三个元素是计算时间
    }
    
    // 每个匹配对包含2帧，总时间主要包含2次描述子提取和1次匹配
    // 估算：描述子提取时间 ≈ (总时间 * 0.9) / (2 * 匹配对数)
    double descriptor_time_ratio = 0.9;  // 描述子提取时间占总时间的比例
    return (total_time * descriptor_time_ratio) / (2.0 * results.size());
}

// 加载带标签的点云（从KITTI格式的.bin文件和.label文件）
bool loadPointCloudWithLabels(const std::string& bin_filename, 
                              const std::string& label_filename,
                              pcl::PointCloud<pcl::PointXYZL>::Ptr& cloud) {
    // 读取KITTI格式的.bin文件
    std::ifstream bin_file(bin_filename, std::ios::binary);
    if (!bin_file.is_open()) {
        std::cerr << "Could not open bin file: " << bin_filename << std::endl;
        return false;
    }
    
    // 读取SemanticKITTI格式的.label文件
    std::ifstream label_file(label_filename, std::ios::binary);
    if (!label_file.is_open()) {
        std::cerr << "Could not open label file: " << label_filename << std::endl;
        bin_file.close();
        return false;
    }
    
    // 获取文件大小
    bin_file.seekg(0, std::ios::end);
    std::streampos bin_file_size = bin_file.tellg();
    bin_file.seekg(0, std::ios::beg);
    
    label_file.seekg(0, std::ios::end);
    std::streampos label_file_size = label_file.tellg();
    label_file.seekg(0, std::ios::beg);
    
    // 每个点包含4个float（x, y, z, 反射强度），这里只使用 xyz；
    // 每个标签是 uint32_t。
    size_t num_points = bin_file_size / (4 * sizeof(float));
    size_t num_labels = label_file_size / sizeof(uint32_t);
    
    if (num_points != num_labels) {
        std::cerr << "Error: Number of points (" << num_points << ") does not match number of labels (" << num_labels << ")" << std::endl;
        bin_file.close();
        label_file.close();
        return false;
    }
    
    cloud->clear();
    cloud->reserve(num_points);
    
    for (size_t i = 0; i < num_points; ++i) {
        float x, y, z;
        bin_file.read(reinterpret_cast<char*>(&x), sizeof(float));
        bin_file.read(reinterpret_cast<char*>(&y), sizeof(float));
        bin_file.read(reinterpret_cast<char*>(&z), sizeof(float));
        bin_file.seekg(sizeof(float), std::ios::cur);
        
        uint32_t label;
        label_file.read(reinterpret_cast<char*>(&label), sizeof(uint32_t));
        
        // 只使用标签的低16位作为语义标签
        uint32_t semantic_label = label & 0xFFFF;
        
        pcl::PointXYZL point;
        point.x = x;
        point.y = y;
        point.z = z;
        point.label = semantic_label;
        cloud->push_back(point);
    }
    
    bin_file.close();
    label_file.close();
    return true;
}

// 从YAML文件读取配置参数
bool loadConfigFromYAML(const std::string& yaml_file, ConfigSetting& config) {
    try {
        YAML::Node yaml_config = YAML::LoadFile(yaml_file);
        
        // 读取配置参数，如果不存在则使用默认值
        if (yaml_config["voxel_size"]) {
            config.voxel_size_ = yaml_config["voxel_size"].as<float>();
        }
        if (yaml_config["plane_detection_thre"]) {
            config.plane_detection_thre_ = yaml_config["plane_detection_thre"].as<float>();
        }
        if (yaml_config["voxel_init_num"]) {
            config.voxel_init_num_ = yaml_config["voxel_init_num"].as<int>();
        }
        if (yaml_config["useful_corner_num"]) {
            config.useful_corner_num_ = yaml_config["useful_corner_num"].as<int>();
        }
        if (yaml_config["proj_plane_num"]) {
            config.proj_plane_num_ = yaml_config["proj_plane_num"].as<int>();
        }
        if (yaml_config["proj_image_resolution"]) {
            config.proj_image_resolution_ = yaml_config["proj_image_resolution"].as<float>();
        }
        if (yaml_config["proj_image_high_inc"]) {
            config.proj_image_high_inc_ = yaml_config["proj_image_high_inc"].as<float>();
        }
        if (yaml_config["proj_dis_min"]) {
            config.proj_dis_min_ = yaml_config["proj_dis_min"].as<float>();
        }
        if (yaml_config["proj_dis_max"]) {
            config.proj_dis_max_ = yaml_config["proj_dis_max"].as<float>();
        }
        if (yaml_config["summary_min_thre"]) {
            config.summary_min_thre_ = yaml_config["summary_min_thre"].as<float>();
        }
        if (yaml_config["descriptor_near_num"]) {
            config.descriptor_near_num_ = yaml_config["descriptor_near_num"].as<float>();
        }
        if (yaml_config["descriptor_min_len"]) {
            config.descriptor_min_len_ = yaml_config["descriptor_min_len"].as<float>();
        }
        if (yaml_config["descriptor_max_len"]) {
            config.descriptor_max_len_ = yaml_config["descriptor_max_len"].as<float>();
        }
        if (yaml_config["non_max_suppression_radius"]) {
            config.non_max_suppression_radius_ = yaml_config["non_max_suppression_radius"].as<float>();
        }
        if (yaml_config["std_side_resolution"]) {
            config.std_side_resolution_ = yaml_config["std_side_resolution"].as<float>();
        }
        if (yaml_config["skip_near_num"]) {
            config.skip_near_num_ = yaml_config["skip_near_num"].as<int>();
        }
        if (yaml_config["candidate_num"]) {
            config.candidate_num_ = yaml_config["candidate_num"].as<int>();
        }
        if (yaml_config["rough_dis_threshold"]) {
            config.rough_dis_threshold_ = yaml_config["rough_dis_threshold"].as<float>();
        }
        if (yaml_config["similarity_threshold"]) {
            config.similarity_threshold_ = yaml_config["similarity_threshold"].as<float>();
        }
        if (yaml_config["icp_threshold"]) {
            config.icp_threshold_ = yaml_config["icp_threshold"].as<float>();
        }
        if (yaml_config["normal_threshold"]) {
            config.normal_threshold_ = yaml_config["normal_threshold"].as<float>();
        }
        if (yaml_config["dis_threshold"]) {
            config.dis_threshold_ = yaml_config["dis_threshold"].as<float>();
        }
        if (yaml_config["plane_merge_normal_thre"]) {
            config.plane_merge_normal_thre_ = yaml_config["plane_merge_normal_thre"].as<float>();
        }
        if (yaml_config["plane_merge_dis_thre"]) {
            config.plane_merge_dis_thre_ = yaml_config["plane_merge_dis_thre"].as<float>();
        }
        if (yaml_config["semantic_vertex_match_threshold"]) {
            config.semantic_vertex_match_threshold_ = yaml_config["semantic_vertex_match_threshold"].as<int>();
        }
        if (yaml_config["semantic_ratio_threshold"]) {
            config.semantic_ratio_threshold_ = yaml_config["semantic_ratio_threshold"].as<float>();
        }
        if (yaml_config["semantic_icp_weight"]) {
            config.semantic_icp_weight_ = yaml_config["semantic_icp_weight"].as<float>();
        }
        
        // 读取屏蔽标签列表（向后兼容）
        if (yaml_config["excluded_labels"]) {
            config.excluded_labels.clear();
            if (yaml_config["excluded_labels"].IsSequence()) {
                // 如果是列表格式
                for (const auto& label_node : yaml_config["excluded_labels"]) {
                    uint32_t label = label_node.as<uint32_t>();
                    config.excluded_labels.insert(label);
                }
            } else if (yaml_config["excluded_labels"].IsScalar()) {
                // 如果是单个值
                uint32_t label = yaml_config["excluded_labels"].as<uint32_t>();
                config.excluded_labels.insert(label);
            }
        }
        
        // 读取平面体素屏蔽标签列表
        if (yaml_config["excluded_labels_plane"]) {
            config.excluded_labels_plane.clear();
            if (yaml_config["excluded_labels_plane"].IsSequence()) {
                for (const auto& label_node : yaml_config["excluded_labels_plane"]) {
                    uint32_t label = label_node.as<uint32_t>();
                    config.excluded_labels_plane.insert(label);
                }
            } else if (yaml_config["excluded_labels_plane"].IsScalar()) {
                uint32_t label = yaml_config["excluded_labels_plane"].as<uint32_t>();
                config.excluded_labels_plane.insert(label);
            }
        }
        
        // Note: excluded_labels_non_plane is not used because BTC descriptor does not process non-plane voxels
        
        return true;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error reading YAML config file: " << e.what() << std::endl;
        return false;
    }
}

// 保存测试指标到txt文件
void saveMetricsToTxt(const std::string& output_file,
                     const std::string& sequence_name,
                     double success_rate,
                     double recall_at_precision1,
                     double avg_descriptor_time,
                     double avg_matching_time,
                     bool is_first_sequence = false) {
    std::ofstream out_file;
    if (is_first_sequence) {
        out_file.open(output_file, std::ios::out);  // 覆盖模式（清空文件）
    } else {
        out_file.open(output_file, std::ios::app);  // 追加模式
    }
    
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << output_file << std::endl;
        return;
    }
    
    out_file << "=== Sequence: " << sequence_name << " ===" << std::endl;
    out_file << "(1) SR (Success Rate): " << success_rate << std::endl;
    out_file << "(2) Recall@Precision=1: " << recall_at_precision1 << std::endl;
    out_file << "(3) Average descriptor extraction time per frame: " << avg_descriptor_time << " ms" << std::endl;
    out_file << "(4) Average matching time per pair: " << avg_matching_time << " ms" << std::endl;
    out_file << std::endl;
    
    out_file.close();
}

// 匹配结果结构（包含得分和位姿变换）
struct MatchResult {
    double score;
    bool has_transform;
    Eigen::Vector3d translation;
    Eigen::Matrix3d rotation;
};

// 计算匹配得分和位姿（使用语义三角形描述子）
MatchResult computeMatchScoreWithLabels(
    SemanticTriangularDescManager& manager,
    const pcl::PointCloud<pcl::PointXYZL>::Ptr& source_cloud,
    const pcl::PointCloud<pcl::PointXYZL>::Ptr& target_cloud,
    double* computation_time_ms) {
    
    MatchResult result;
    result.score = 0.0;
    result.has_transform = false;
    result.translation = Eigen::Vector3d::Zero();
    result.rotation = Eigen::Matrix3d::Identity();
    
    if (source_cloud->empty() || target_cloud->empty()) {
        if (computation_time_ms) {
            *computation_time_ms = 0.0;
        }
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // 清理数据库和平面点云（确保每次匹配独立）
    manager.data_base_.clear();
    manager.plane_cloud_vec_.clear();
    
    // 生成目标点云的语义三角形描述子（先添加到数据库，frame_id=0）
    std::vector<SemanticTriangularDescriptor> target_stds;
    manager.GenerateSemanticTriangularDescs(target_cloud, 0, target_stds);
    // 将目标描述子添加到数据库
    manager.AddSemanticTriangularDescs(target_stds);
    
    // 生成源点云的语义三角形描述子（用于查询，frame_id=1）
    std::vector<SemanticTriangularDescriptor> source_stds;
    manager.GenerateSemanticTriangularDescs(source_cloud, 1, source_stds);
    
    // 如果描述子数量太少，返回0
    if (source_stds.empty() || target_stds.empty()) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        if (computation_time_ms) {
            *computation_time_ms = duration.count();
        }
        return result;
    }
    
    // 保存原始阈值，临时设为0以获取所有得分（与原版一致）
    double original_threshold = manager.config_setting_.icp_threshold_;
    manager.config_setting_.icp_threshold_ = 0.0;
    
    // 搜索匹配
    std::pair<int, double> loop_result;
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
    std::vector<std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor>> loop_std_pair;
    
    manager.SearchLoop(source_stds, loop_result, loop_transform, loop_std_pair);
    
    // 恢复原始阈值
    manager.config_setting_.icp_threshold_ = original_threshold;
    
    // 匹配得分和位姿
    result.score = loop_result.second;
    if (loop_result.first >= 0 && result.score > 0) {
        result.has_transform = true;
        result.translation = loop_transform.first;
        result.rotation = loop_transform.second;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    if (computation_time_ms) {
        *computation_time_ms = duration.count();
    }
    
    return result;
}

// Transform pose from point cloud coordinate system to KITTI pose coordinate system
// Coordinate transformation: (x_new, y_new, z_new) = (-y_old, -z_old, x_old)
// This transformation matrix converts from point cloud coords to KITTI ground truth coords
static void transform_pose_for_kitti(Eigen::Matrix4d& T) {
    // Transformation matrix: [x_new, y_new, z_new]^T = T_swap * [x_old, y_old, z_old]^T
    // [x_new]   [0  -1   0] [x_old]
    // [y_new] = [0   0  -1] [y_old]
    // [z_new]   [1   0   0] [z_old]
    Eigen::Matrix3d T_swap;
    T_swap << 0, -1, 0,
              0, 0, -1,
              1, 0, 0;
    
    // Extract rotation and translation
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);
    
    // Transform rotation: R_new = T_swap * R * T_swap^T
    Eigen::Matrix3d R_transformed = T_swap * R * T_swap.transpose();
    
    // Transform translation: [x_new, y_new, z_new] = [-y_old, -z_old, x_old]
    Eigen::Vector3d t_transformed(-t.y(), -t.z(), t.x());
    
    // Update transformation matrix
    T.block<3, 3>(0, 0) = R_transformed;
    T.block<3, 1>(0, 3) = t_transformed;
}

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8 && argc != 9) {
        std::cout << "Usage: " << argv[0] 
                  << " <input_bin_folder> <target_bin_folder> <poses_folder> <labels_folder> <output_folder> <config_yaml> [pose_distance_threshold] [is_kitti]" << std::endl;
        std::cout << "  input_bin_folder: Folder containing input .bin point cloud files (with sequence subfolders)" << std::endl;
        std::cout << "  target_bin_folder: Folder containing target .bin point cloud files (with sequence subfolders)" << std::endl;
        std::cout << "  poses_folder: Folder containing pose .txt files (one per sequence)" << std::endl;
        std::cout << "  labels_folder: Folder containing .label semantic files (with sequence subfolders)" << std::endl;
        std::cout << "  output_folder: Folder to save evaluation results" << std::endl;
        std::cout << "  config_yaml: Path to YAML configuration file" << std::endl;
        std::cout << "  pose_distance_threshold: Maximum pose distance for true match (default: 20.0 meters)" << std::endl;
        std::cout << "  is_kitti: 1 for KITTI dataset (apply coordinate transform), 0 for others (default: 0)" << std::endl;
        return -1;
    }

    std::string input_bin_folder = argv[1];
    std::string target_bin_folder = argv[2];
    std::string poses_folder = argv[3];
    std::string labels_folder = argv[4];
    std::string output_folder = argv[5];
    std::string config_yaml = argv[6];
    double pose_distance_threshold = (argc > 7) ? std::stod(argv[7]) : 20.0;  // 默认20.0米
    bool is_kitti = (argc > 8) ? (std::stoi(argv[8]) != 0) : false;  // 默认false

    // 创建输出文件夹
    fs::create_directories(output_folder);

    // 获取所有子文件夹
    std::vector<std::string> input_subfolders = getSubfolders(input_bin_folder);
    std::vector<std::string> target_subfolders = getSubfolders(target_bin_folder);

    // 从YAML文件读取配置参数
    ConfigSetting config;  // 使用默认值初始化
    if (!loadConfigFromYAML(config_yaml, config)) {
        std::cerr << "Warning: Failed to load config from " << config_yaml 
                  << ", using default values." << std::endl;
    } else {
        std::cout << "Successfully loaded configuration from " << config_yaml << std::endl;
    }
    
    // 输出当前配置参数（用于验证）
    std::cout << "Configuration parameters:" << std::endl;
    std::cout << "  is_kitti: " << (is_kitti ? "true (will apply coordinate transform)" : "false") << std::endl;
    std::cout << "  voxel_size: " << config.voxel_size_ << std::endl;
    std::cout << "  plane_detection_thre: " << config.plane_detection_thre_ << std::endl;
    std::cout << "  voxel_init_num: " << config.voxel_init_num_ << std::endl;
    std::cout << "  useful_corner_num: " << config.useful_corner_num_ << std::endl;
    std::cout << "  descriptor_min_len: " << config.descriptor_min_len_ << std::endl;
    std::cout << "  descriptor_max_len: " << config.descriptor_max_len_ << std::endl;
    std::cout << "  std_side_resolution: " << config.std_side_resolution_ << std::endl;
    std::cout << "  rough_dis_threshold: " << config.rough_dis_threshold_ << std::endl;
    std::cout << "  similarity_threshold: " << config.similarity_threshold_ << std::endl;
    std::cout << "  icp_threshold: " << config.icp_threshold_ << std::endl;
    
    SemanticTriangularDescManager manager(config);
    manager.print_debug_info_ = false;  // 关闭调试信息以加快速度
    
    std::cout << "Pose distance threshold for true match: " << pose_distance_threshold << " meters" << std::endl;
    
    // 处理所有序列的点云对
    for (size_t i = 0; i < input_subfolders.size(); ++i) {
        std::string input_subfolder = input_subfolders[i];
        std::string target_subfolder = target_subfolders[i];
        std::string sequence_name = fs::path(input_subfolder).filename().string();
        bool is_first_sequence = (i == 0);
        
        std::cout << "Processing sequence: " << sequence_name << std::endl;

        // 读取该序列位姿文件
        std::string pose_file = (fs::path(poses_folder) / (sequence_name + ".txt")).string();
        auto poses = readKittiPoses(pose_file);

        // 获取当前序列的所有点云文件
        std::vector<std::string> input_bin_files = getBinFiles(input_subfolder);
        std::vector<std::string> target_bin_files = getBinFiles(target_subfolder);
        
        // 获取对应的标签文件文件夹
        std::string labels_subfolder = (fs::path(labels_folder) / sequence_name).string();
        std::vector<std::string> label_files = getLabelFiles(labels_subfolder);

        std::vector<std::tuple<double, bool, double>> sequence_results;  // 用于计算F1-score
        std::vector<DetailedMatchResult> detailed_results;  // 用于保存详细结果

        // 计算所有点云对之间的匹配得分（只在相同序列内匹配）
        for (const auto& input_bin_file : input_bin_files) {
            int input_num = extractNumberFromFilename(input_bin_file);
            
            // 加载输入点云和标签
            std::string input_label_file = (fs::path(labels_subfolder) / 
                                           (fs::path(input_bin_file).stem().string() + ".label")).string();
            pcl::PointCloud<pcl::PointXYZL>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZL>);
            if (!loadPointCloudWithLabels(input_bin_file, input_label_file, input_cloud)) {
                std::cerr << "Failed to load input cloud with labels: " << input_bin_file << std::endl;
                continue;
            }

            for (const auto& target_bin_file : target_bin_files) {
                int target_num = extractNumberFromFilename(target_bin_file);
                
                // 加载目标点云和标签
                std::string target_label_file = (fs::path(labels_subfolder) / 
                                               (fs::path(target_bin_file).stem().string() + ".label")).string();
                pcl::PointCloud<pcl::PointXYZL>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZL>);
                if (!loadPointCloudWithLabels(target_bin_file, target_label_file, target_cloud)) {
                    std::cerr << "Failed to load target cloud with labels: " << target_bin_file << std::endl;
                    continue;
                }

                double computation_time = 0.0;
                MatchResult match_result = computeMatchScoreWithLabels(manager, input_cloud, target_cloud, &computation_time);
                bool is_true = false;
                double pose_dist = -1.0;  // 默认值，表示无效距离
                if (input_num >= 0 && static_cast<size_t>(input_num) < poses.size() &&
                    target_num >= 0 && static_cast<size_t>(target_num) < poses.size() &&
                    poses[input_num].valid && poses[target_num].valid) {
                    pose_dist = distance3(poses[input_num], poses[target_num]);
                    is_true = pose_dist < pose_distance_threshold;
                }
                
                // 保存用于计算F1-score的结果
                sequence_results.push_back({match_result.score, is_true, computation_time});
                
                // 保存详细结果（包含位姿）
                DetailedMatchResult detailed_result;
                detailed_result.input_id = input_num;
                detailed_result.target_id = target_num;
                detailed_result.match_score = match_result.score;
                detailed_result.is_true_match = is_true;
                detailed_result.computation_time = computation_time;
                detailed_result.pose_distance = pose_dist;
                detailed_result.has_transform = match_result.has_transform;
                detailed_result.translation = match_result.translation;
                detailed_result.rotation = match_result.rotation;
                detailed_results.push_back(detailed_result);
            }
        }

        // 找到F1-score最大的阈值
        std::vector<double> thresholds;
        for (const auto& result : sequence_results) {
            thresholds.push_back(std::get<0>(result));
        }
        std::sort(thresholds.begin(), thresholds.end());
        thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
        
        double best_f1 = 0.0;
        double best_threshold = 0.0;
        
        for (double threshold : thresholds) {
            std::vector<std::pair<double, bool>> threshold_results;
            for (const auto& result : sequence_results) {
                bool predicted_match = (std::get<0>(result) >= threshold);
                bool actual_match = std::get<1>(result);
                threshold_results.push_back({predicted_match, actual_match});
        }
        
            auto [precision, recall] = calculateMetrics(threshold_results);
            double f1 = (precision + recall == 0.0) ? 0.0 :
                2.0 * precision * recall / (precision + recall);
            
            if (f1 > best_f1) {
                best_f1 = f1;
                best_threshold = threshold;
        }
        }
        
        std::cout << "Sequence " << sequence_name << ":" << std::endl;
        std::cout << "  Best F1-score: " << best_f1 << " at threshold: " << best_threshold << std::endl;
        
        // 在最佳阈值下，找出所有TP匹配并计算位姿RMSE
        std::vector<DetailedMatchResult> tp_matches;
        for (const auto& result : detailed_results) {
            if (result.match_score >= best_threshold && result.is_true_match && result.has_transform) {
                tp_matches.push_back(result);
            }
        }
        
        std::cout << "  True Positive matches with valid pose: " << tp_matches.size() << std::endl;
        
        if (tp_matches.empty()) {
            std::cout << "  No TP matches with valid pose, skipping RMSE calculation." << std::endl;
        } else {
            // 计算位姿估计误差（相对位姿误差，包括平移和旋转）
            // 保存所有误差（包含超出范围的）
            std::vector<double> translation_errors_all;
            std::vector<double> rotation_errors_all;
            std::vector<std::pair<int, int>> match_indices_all;  // 保存匹配对的索引
            
            // 保存过滤后的误差（不包含超出范围的）
            std::vector<double> translation_errors;
            std::vector<double> rotation_errors;
            std::vector<std::pair<int, int>> match_indices;  // 保存匹配对的索引
            
            int discarded_count = 0;  // 统计被舍弃的样本数
            
            for (const auto& match : tp_matches) {
                int input_id = match.input_id;
                int target_id = match.target_id;
                
                if (input_id >= 0 && static_cast<size_t>(input_id) < poses.size() &&
                    target_id >= 0 && static_cast<size_t>(target_id) < poses.size() &&
                    poses[input_id].valid && poses[target_id].valid) {
                    
                    // 计算真值相对位姿（从 input 到 target）
                    Eigen::Matrix4d T_input = pose_to_matrix(poses[input_id]);
                    Eigen::Matrix4d T_target = pose_to_matrix(poses[target_id]);
                    Eigen::Matrix4d T_gt_rel = compute_relative_pose(T_input, T_target);
                    
                    // 构建估计的相对位姿（从 loop_transform）
                    // 注意：SearchLoop返回的transform是从source（input）到target的变换
                    Eigen::Matrix4d T_est_rel = Eigen::Matrix4d::Identity();
                    T_est_rel.block<3, 3>(0, 0) = match.rotation;  // Rotation
                    T_est_rel.block<3, 1>(0, 3) = match.translation;  // Translation
                    
                    // 对于KITTI数据集，需要对估计位姿进行坐标轴变换（交换x和z轴）
                    // 因为KITTI数据集的真值位姿坐标系与点云坐标系不一致
                    if (is_kitti) {
                        transform_pose_for_kitti(T_est_rel);
                    }
                    
                    // 计算位姿误差
                    double trans_error, rot_error;
                    compute_pose_error(T_est_rel, T_gt_rel, trans_error, rot_error);
        
                    // 计算真值相对位姿的大小（平移距离和旋转角度）
                    Eigen::Vector3d t_gt = T_gt_rel.block<3, 1>(0, 3);
                    double gt_translation_norm = t_gt.norm();
                    
                    Eigen::Matrix3d R_gt = T_gt_rel.block<3, 3>(0, 0);
                    Eigen::AngleAxisd gt_angle_axis(R_gt);
                    double gt_rotation_angle = std::abs(gt_angle_axis.angle()) * 180.0 / M_PI;
                    
                    // 保存所有误差（包含超出范围的）
                    translation_errors_all.push_back(trans_error);
                    rotation_errors_all.push_back(rot_error);
                    match_indices_all.push_back({input_id, target_id});
                    
                    // 只有当误差小于位姿本身时才纳入过滤后的计算
                    if (trans_error < gt_translation_norm && rot_error < gt_rotation_angle) {
                        translation_errors.push_back(trans_error);
                        rotation_errors.push_back(rot_error);
                        match_indices.push_back({input_id, target_id});
                    } else {
                        discarded_count++;
                    }
                }
            }
            
            if (discarded_count > 0) {
                std::cout << "  Discarded " << discarded_count << " samples with error larger than pose magnitude" << std::endl;
            }
            
            // 保存所有误差（包含超出范围的）到CSV文件
            if (!translation_errors_all.empty()) {
                std::string pose_error_csv_all_file = (fs::path(output_folder) / (sequence_name + "_pose_errors_all.csv")).generic_string();
                std::ofstream pose_error_csv_all(pose_error_csv_all_file);
                pose_error_csv_all << "query_idx,database_idx,translation_error(m),rotation_error(deg)" << std::endl;
                for (size_t i = 0; i < translation_errors_all.size(); i++) {
                    pose_error_csv_all << match_indices_all[i].first << "," << match_indices_all[i].second << ","
                                       << std::fixed << std::setprecision(6) << translation_errors_all[i] << ","
                                       << rotation_errors_all[i] << std::endl;
                }
                pose_error_csv_all.close();
                std::cout << "  Saved all pose errors (including out-of-range) to " << pose_error_csv_all_file << std::endl;
            }
            
            // 保存过滤后的误差（不包含超出范围的）到CSV文件
            if (!translation_errors.empty()) {
                std::string pose_error_csv_file = (fs::path(output_folder) / (sequence_name + "_pose_errors.csv")).generic_string();
                std::ofstream pose_error_csv(pose_error_csv_file);
                pose_error_csv << "query_idx,database_idx,translation_error(m),rotation_error(deg)" << std::endl;
                for (size_t i = 0; i < translation_errors.size(); i++) {
                    pose_error_csv << match_indices[i].first << "," << match_indices[i].second << ","
                                   << std::fixed << std::setprecision(6) << translation_errors[i] << ","
                                   << rotation_errors[i] << std::endl;
                }
                pose_error_csv.close();
                std::cout << "  Saved filtered pose errors (excluding out-of-range) to " << pose_error_csv_file << std::endl;
            }
            
            // 计算所有误差的RMSE（包含超出范围的）
            double translation_rmse_all = 0.0;
            double translation_mean_all = 0.0;
            double trans_max_all = 0.0;
            double rotation_rmse_all = 0.0;
            double rotation_mean_all = 0.0;
            double rot_max_all = 0.0;
            
            if (!translation_errors_all.empty()) {
                double trans_squared_sum_all = 0.0;
                double trans_sum_all = 0.0;
                for (double err : translation_errors_all) {
                    trans_squared_sum_all += err * err;
                    trans_sum_all += err;
                    if (err > trans_max_all) trans_max_all = err;
                }
                translation_rmse_all = std::sqrt(trans_squared_sum_all / translation_errors_all.size());
                translation_mean_all = trans_sum_all / translation_errors_all.size();
                
                double rot_squared_sum_all = 0.0;
                double rot_sum_all = 0.0;
                for (double err : rotation_errors_all) {
                    rot_squared_sum_all += err * err;
                    rot_sum_all += err;
                    if (err > rot_max_all) rot_max_all = err;
                }
                rotation_rmse_all = std::sqrt(rot_squared_sum_all / rotation_errors_all.size());
                rotation_mean_all = rot_sum_all / rotation_errors_all.size();
                
                std::cout << "  [All Errors] Translation Error RMSE: " << translation_rmse_all << " m" << std::endl;
                std::cout << "  [All Errors] Translation Error Mean: " << translation_mean_all << " m" << std::endl;
                std::cout << "  [All Errors] Translation Error Max: " << trans_max_all << " m" << std::endl;
                std::cout << "  [All Errors] Rotation Error RMSE: " << rotation_rmse_all << " deg" << std::endl;
                std::cout << "  [All Errors] Rotation Error Mean: " << rotation_mean_all << " deg" << std::endl;
                std::cout << "  [All Errors] Rotation Error Max: " << rot_max_all << " deg" << std::endl;
            }
            
            // 计算过滤后误差的RMSE（不包含超出范围的）
            double translation_rmse = 0.0;
            double translation_mean = 0.0;
            double trans_max = 0.0;
            double rotation_rmse = 0.0;
            double rotation_mean = 0.0;
            double rot_max = 0.0;
            
            if (!translation_errors.empty()) {
                double trans_squared_sum = 0.0;
                double trans_sum = 0.0;
                for (double err : translation_errors) {
                    trans_squared_sum += err * err;
                    trans_sum += err;
                    if (err > trans_max) trans_max = err;
                }
                translation_rmse = std::sqrt(trans_squared_sum / translation_errors.size());
                translation_mean = trans_sum / translation_errors.size();
                
                double rot_squared_sum = 0.0;
                double rot_sum = 0.0;
                for (double err : rotation_errors) {
                    rot_squared_sum += err * err;
                    rot_sum += err;
                    if (err > rot_max) rot_max = err;
                }
                rotation_rmse = std::sqrt(rot_squared_sum / rotation_errors.size());
                rotation_mean = rot_sum / rotation_errors.size();
                
                std::cout << "  [Filtered Errors] Translation Error RMSE: " << translation_rmse << " m" << std::endl;
                std::cout << "  [Filtered Errors] Translation Error Mean: " << translation_mean << " m" << std::endl;
                std::cout << "  [Filtered Errors] Translation Error Max: " << trans_max << " m" << std::endl;
                std::cout << "  [Filtered Errors] Rotation Error RMSE: " << rotation_rmse << " deg" << std::endl;
                std::cout << "  [Filtered Errors] Rotation Error Mean: " << rotation_mean << " deg" << std::endl;
                std::cout << "  [Filtered Errors] Rotation Error Max: " << rot_max << " deg" << std::endl;
            }
            
            // 保存RMSE结果到TXT文件（无论是否有数据，都保存文件）
            std::string rmse_output_file = (fs::path(output_folder) / (sequence_name + "_pose_metrics.txt")).generic_string();
            std::ofstream rmse_file(rmse_output_file);
            rmse_file << std::fixed << std::setprecision(6);
            rmse_file << "Sequence: " << sequence_name << std::endl;
            rmse_file << "========================================" << std::endl;
            rmse_file << "F1-Optimal Threshold Selection:" << std::endl;
            rmse_file << "  F1-Optimal Threshold: " << best_threshold << std::endl;
            rmse_file << "  F1-Optimal Score: " << best_f1 << std::endl;
            rmse_file << "  Number of TP matches: " << tp_matches.size() << std::endl;
            rmse_file << "========================================" << std::endl;
            rmse_file << "Pose Estimation Error Statistics - ALL ERRORS (including outliers):" << std::endl;
            if (!translation_errors_all.empty()) {
                rmse_file << "  Translation Error RMSE: " << translation_rmse_all << " m" << std::endl;
                rmse_file << "  Translation Error Mean: " << translation_mean_all << " m" << std::endl;
                rmse_file << "  Translation Error Max: " << trans_max_all << " m" << std::endl;
                rmse_file << "  Rotation Error RMSE: " << rotation_rmse_all << " deg" << std::endl;
                rmse_file << "  Rotation Error Mean: " << rotation_mean_all << " deg" << std::endl;
                rmse_file << "  Rotation Error Max: " << rot_max_all << " deg" << std::endl;
                rmse_file << "  Total TP matches with valid pose estimates: " << translation_errors_all.size() << std::endl;
            } else {
                rmse_file << "  No valid pose pairs for RMSE calculation (all samples are outliers or invalid)." << std::endl;
            }
            rmse_file << "========================================" << std::endl;
            rmse_file << "Pose Estimation Error Statistics - FILTERED ERRORS (excluding outliers):" << std::endl;
            if (!translation_errors.empty()) {
                rmse_file << "  Translation Error RMSE: " << translation_rmse << " m" << std::endl;
                rmse_file << "  Translation Error Mean: " << translation_mean << " m" << std::endl;
                rmse_file << "  Translation Error Max: " << trans_max << " m" << std::endl;
                rmse_file << "  Rotation Error RMSE: " << rotation_rmse << " deg" << std::endl;
                rmse_file << "  Rotation Error Mean: " << rotation_mean << " deg" << std::endl;
                rmse_file << "  Rotation Error Max: " << rot_max << " deg" << std::endl;
                rmse_file << "  Total TP matches with valid pose estimates: " << translation_errors.size() << std::endl;
                rmse_file << "  Discarded samples (outliers): " << discarded_count << std::endl;
            } else {
                rmse_file << "  No valid pose pairs after filtering (all samples are outliers)." << std::endl;
                rmse_file << "  Discarded samples (outliers): " << discarded_count << std::endl;
    }
            rmse_file << "========================================" << std::endl;
            rmse_file.close();
            
            std::cout << "  Saved pose metrics to " << rmse_output_file << std::endl;
            
            if (translation_errors_all.empty()) {
                std::cout << "  Warning: No valid pose pairs for RMSE calculation (all samples are outliers or invalid)." << std::endl;
            }
        }
    }

    std::cout << "Pose estimation evaluation completed. Results saved to: " << output_folder << std::endl;

    return 0;
}

#ifndef SEMANTIC_ICP_H
#define SEMANTIC_ICP_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/correspondence_estimation.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <filesystem>
#include <type_traits>

// 語義點雲類型定義：基於PointXYZINormal，添加語義標籤
struct PointXYZINormalLabel
{
    PCL_ADD_POINT4D;
    PCL_ADD_NORMAL4D;
    float intensity;
    uint32_t label;  // 語義標籤（低16位為語義標籤）
    float curvature; // 時間戳
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZINormalLabel,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, normal_x, normal_x)
    (float, normal_y, normal_y)
    (float, normal_z, normal_z)
    (float, intensity, intensity)
    (uint32_t, label, label)
    (float, curvature, curvature)
)

// Ceres優化成本函數：語義感知的點對點距離
struct SemanticPointToPointCost {
    SemanticPointToPointCost(const Eigen::Vector3d& source_point, 
                            const Eigen::Vector3d& target_point,
                            uint32_t source_label,
                            uint32_t target_label,
                            double semantic_weight)
        : source_point_(source_point), target_point_(target_point),
          source_label_(source_label), target_label_(target_label),
          semantic_weight_(semantic_weight) {}

    template <typename T>
    bool operator()(const T* const rotation, const T* const translation, T* residuals) const {
        // 將源點轉換為T類型
        T source_point[3];
        for (int i = 0; i < 3; ++i) {
            source_point[i] = T(source_point_[i]);
        }
        
        // 應用旋轉變換
        T rotated_point[3];
        ceres::AngleAxisRotatePoint(rotation, source_point, rotated_point);
        
        // 應用平移變換
        T transformed_point[3];
        for (int i = 0; i < 3; ++i) {
            transformed_point[i] = rotated_point[i] + translation[i];
        }
        
        // 計算幾何距離
        T geometric_distance = T(0.0);
        for (int i = 0; i < 3; ++i) {
            T diff = transformed_point[i] - T(target_point_[i]);
            geometric_distance += diff * diff;
        }
        geometric_distance = ceres::sqrt(geometric_distance);
        
        // 計算語義誤差
        T semantic_error = T(0.0);
        if (source_label_ != target_label_) {
            semantic_error = T(semantic_weight_);
        }
        
        // 總誤差 = 幾何距離 + 語義誤差
        residuals[0] = geometric_distance + semantic_error;
        
        return true;
    }

private:
    const Eigen::Vector3d source_point_;
    const Eigen::Vector3d target_point_;
    const uint32_t source_label_;
    const uint32_t target_label_;
    const double semantic_weight_;
};

// 輔助函數：獲取點的語義標籤
template<typename PointT>
uint32_t getPointLabel(const PointT& point) {
    return 0;  // 默認返回0
}

// 特化版本：PointXYZINormalLabel
template<>
uint32_t getPointLabel<PointXYZINormalLabel>(const PointXYZINormalLabel& point) {
    return point.label & 0xFFFF;
}

// 語義感知的對應關係估計器
class SemanticCorrespondenceEstimation {
public:
    SemanticCorrespondenceEstimation(double semantic_weight = 1.0) 
        : semantic_weight_(semantic_weight) {}
    
    void setSemanticWeight(double weight) { semantic_weight_ = weight; }
    
    // 計算語義感知的對應關係
    template<typename PointT>
    void determineCorrespondences(
        const typename pcl::PointCloud<PointT>::ConstPtr& source,
        const typename pcl::PointCloud<PointT>::ConstPtr& target,
        pcl::Correspondences& correspondences,
        double max_distance = 2.0) {
        
        correspondences.clear();
        
        // 構建目標點雲的KD樹
        pcl::search::KdTree<PointT> kdtree;
        kdtree.setInputCloud(target);
        
        std::vector<int> nn_indices(1);
        std::vector<float> nn_distances(1);
        
        for (size_t i = 0; i < source->size(); ++i) {
            const auto& src_point = source->points[i];
            
            // 在目標點雲中搜索最近鄰
            if (kdtree.nearestKSearch(src_point, 1, nn_indices, nn_distances) > 0) {
                double distance = std::sqrt(nn_distances[0]);
                
                if (distance < max_distance) {
                    const auto& tgt_point = target->points[nn_indices[0]];
                    
                    // 獲取語義標籤
                    uint32_t src_label = getPointLabel(src_point);
                    uint32_t tgt_label = getPointLabel(tgt_point);
                    
                    bool semantic_same = (src_label == tgt_label);
                    
                    // 計算語義誤差：語義相同為0，不同為semantic_weight
                    double semantic_error = semantic_same ? 0.0 : semantic_weight_;
                    
                    // 總距離 = 幾何距離 + 語義誤差
                    double total_distance = distance + semantic_error;
                    
                    // 根據總距離決定是否接受匹配
                    bool accept_match = false;
                    if (semantic_same) {
                        accept_match = true;  // 語義相同，直接接受
                    } else if (semantic_weight_ < 1.0) {
                        accept_match = true;  // 語義權重較小，允許不同語義匹配
                    }
                    
                    if (accept_match) {
                        pcl::Correspondence corr;
                        corr.index_query = i;
                        corr.index_match = nn_indices[0];
                        corr.distance = total_distance;  // 使用總距離（幾何+語義）
                        correspondences.push_back(corr);
                    }
                }
            }
        }
    }

private:
    double semantic_weight_;
};

// 語義感知的Point-to-Point ICP
template<typename PointT>
Eigen::Matrix4f runSemanticPointToPointICP(
    const typename pcl::PointCloud<PointT>::Ptr& src_cloud,
    const typename pcl::PointCloud<PointT>::Ptr& tgt_cloud,
    double semantic_weight = 0.5,
    int max_iterations = 30) {
    
    // 創建語義感知的對應關係估計器
    SemanticCorrespondenceEstimation semantic_corr_est(semantic_weight);
    
    // 初始位姿為單位矩陣
    Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();
    
    // 迭代優化
    const double convergence_threshold = 1e-6;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // 應用當前位姿到源點雲
        typename pcl::PointCloud<PointT>::Ptr src_transformed(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*src_cloud, *src_transformed, current_pose);
        
        // 計算語義感知的對應關係
        pcl::Correspondences correspondences;
        semantic_corr_est.template determineCorrespondences<PointT>(src_transformed, tgt_cloud, correspondences, 2.0);
        
        if (correspondences.size() < 3) {
            std::cout << "Semantic ICP: Not enough correspondences at iteration " << iter << std::endl;
            break;
        }
        
        // 使用對應關係計算變換
        Eigen::Matrix4f incremental_transform = Eigen::Matrix4f::Identity();
        
        // 構建對應點對
        std::vector<Eigen::Vector3d> src_points, tgt_points;
        for (const auto& corr : correspondences) {
            const auto& src_pt = src_transformed->points[corr.index_query];
            const auto& tgt_pt = tgt_cloud->points[corr.index_match];
            
            src_points.emplace_back(src_pt.x, src_pt.y, src_pt.z);
            tgt_points.emplace_back(tgt_pt.x, tgt_pt.y, tgt_pt.z);
        }
        
        // 使用SVD計算變換矩陣
        if (src_points.size() >= 3) {
            // 計算質心
            Eigen::Vector3d src_centroid = Eigen::Vector3d::Zero();
            Eigen::Vector3d tgt_centroid = Eigen::Vector3d::Zero();
            
            for (size_t i = 0; i < src_points.size(); ++i) {
                src_centroid += src_points[i];
                tgt_centroid += tgt_points[i];
            }
            src_centroid /= src_points.size();
            tgt_centroid /= tgt_points.size();
            
            // 計算協方差矩陣
            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            for (size_t i = 0; i < src_points.size(); ++i) {
                Eigen::Vector3d src_centered = src_points[i] - src_centroid;
                Eigen::Vector3d tgt_centered = tgt_points[i] - tgt_centroid;
                H += src_centered * tgt_centered.transpose();
            }
            
            // SVD分解
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
            if (R.determinant() < 0) {
                Eigen::Matrix3d V = svd.matrixV();
                V.col(2) *= -1;
                R = V * svd.matrixU().transpose();
            }
            
            Eigen::Vector3d t = tgt_centroid - R * src_centroid;
            
            incremental_transform.block<3,3>(0,0) = R.cast<float>();
            incremental_transform.block<3,1>(0,3) = t.cast<float>();
            
            // 更新當前位姿
            current_pose = incremental_transform * current_pose;
            
            // 檢查收斂性
            double translation_norm = t.norm();
            if (translation_norm < convergence_threshold) {
                std::cout << "Semantic ICP converged at iteration " << iter 
                          << " (translation norm: " << translation_norm << ")" << std::endl;
                break;
            }
        }
    }
    
    return current_pose;
}

// 從SemanticKITTI格式的.label文件讀取標籤
bool loadSemanticLabels(const std::string& label_file, std::vector<uint32_t>& labels) {
    std::ifstream label_stream(label_file, std::ios::binary);
    if (!label_stream.is_open()) {
        std::cerr << "Could not open label file: " << label_file << std::endl;
        return false;
    }
    
    // 獲取文件大小
    label_stream.seekg(0, std::ios::end);
    std::streampos label_file_size = label_stream.tellg();
    label_stream.seekg(0, std::ios::beg);
    
    // 每個標籤是uint32_t
    size_t num_labels = label_file_size / sizeof(uint32_t);
    
    labels.clear();
    labels.reserve(num_labels);
    
    for (size_t i = 0; i < num_labels; ++i) {
        uint32_t label;
        label_stream.read(reinterpret_cast<char*>(&label), sizeof(uint32_t));
        // 只使用標籤的低16位作為語義標籤（與SemanticKITTI標準一致）
        uint32_t semantic_label = label & 0xFFFF;
        labels.push_back(semantic_label);
    }
    
    label_stream.close();
    return true;
}

// KITTI velodyne .bin：與 merge_semantic_pt.py 一致，float32 reshape(-1,4)，每行前 3 列為 xyz
inline bool loadKittiVelodyneBin(const std::string& bin_file, std::vector<std::array<float, 3>>& out_xyz) {
    std::ifstream f(bin_file, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Could not open velodyne bin: " << bin_file << std::endl;
        return false;
    }
    f.seekg(0, std::ios::end);
    const std::streampos end = f.tellg();
    f.seekg(0, std::ios::beg);
    const auto bytes = static_cast<size_t>(end);
    if (bytes % (sizeof(float) * 4) != 0) {
        std::cerr << "Invalid velodyne bin size: " << bin_file << std::endl;
        return false;
    }
    const size_t n = bytes / (sizeof(float) * 4);
    out_xyz.resize(n);
    std::vector<float> buf(n * 4);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n * 4 * sizeof(float)));
    for (size_t i = 0; i < n; ++i) {
        out_xyz[i][0] = buf[i * 4 + 0];
        out_xyz[i][1] = buf[i * 4 + 1];
        out_xyz[i][2] = buf[i * 4 + 2];
    }
    return true;
}

// 將語義標籤添加到點雲中
template<typename PointT>
void addSemanticLabelsToCloud(
    typename pcl::PointCloud<PointT>::Ptr& cloud,
    const std::vector<uint32_t>& labels) {
    
    if (cloud->size() != labels.size()) {
        std::cerr << "Error: Point cloud size (" << cloud->size() 
                  << ") does not match label size (" << labels.size() << ")" << std::endl;
        return;
    }
    
    // 如果點類型支持label字段，則添加標籤
    if (std::is_same<PointT, PointXYZINormalLabel>::value) {
        for (size_t i = 0; i < cloud->size(); ++i) {
            cloud->points[i].label = labels[i];
        }
    }
}

// 注意：PCL模板類的顯式實例化需要在cpp文件中進行
// 見laserMapping.cpp文件末尾

#endif // SEMANTIC_ICP_H

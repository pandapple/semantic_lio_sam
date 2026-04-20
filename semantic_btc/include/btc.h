#ifndef SEMANTIC_TRIANGULAR_DESCRIPTOR_H
#define SEMANTIC_TRIANGULAR_DESCRIPTOR_H
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl/common/io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <unordered_set>
#include <ros/publisher.h>
#include <ros/ros.h>
#include <stdio.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <execution>
#include <fstream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <unordered_map>

#define HASH_P 116101
#define MAX_N 10000000000

typedef struct ConfigSetting {
  /* for submap process*/
  double cloud_ds_size_ = 0.25;

  /* for binary descriptor*/
  int useful_corner_num_ = 30;
  float plane_merge_normal_thre_;
  float plane_merge_dis_thre_;
  float plane_detection_thre_ = 0.01;
  float voxel_size_ = 1.0;
  int voxel_init_num_ = 10;
  int proj_plane_num_ = 1;
  float proj_image_resolution_ = 0.5;
  float proj_image_high_inc_ = 0.5;
  float proj_dis_min_ = 0;
  float proj_dis_max_ = 5;
  float summary_min_thre_ = 10;
  int line_filter_enable_ = 0;

  /* for triangle descriptor */
  float descriptor_near_num_ = 10;
  float descriptor_min_len_ = 1;
  float descriptor_max_len_ = 10;
  float non_max_suppression_radius_ = 3.0;
  float std_side_resolution_ = 0.2;

  /* for place recognition*/
  int skip_near_num_ = 20;
  int candidate_num_ = 50;
  int sub_frame_num_ = 10;
  float rough_dis_threshold_ = 0.03;
  float similarity_threshold_ = 0.7;
  float icp_threshold_ = 0.5;
  float normal_threshold_ = 0.1;
  float dis_threshold_ = 0.3;

  /* extrinsic for lidar to vehicle*/
  Eigen::Matrix3d rot_lidar_to_vehicle_;
  Eigen::Vector3d t_lidar_to_vehicle_;

  /* for gt file style*/
  int gt_file_style_ = 0;

  /* for semantic label filtering */
  std::unordered_set<uint32_t> excluded_labels;  // 通用屏蔽标签列表
  std::unordered_set<uint32_t> excluded_labels_plane;  // 平面体素专用屏蔽标签列表
  // Note: BTC descriptor does not use non-plane voxels, so excluded_labels_non_plane is not needed

  /* for descriptor type and semantic matching */
  int semantic_vertex_match_threshold_ = 0;  // 0=使用二进制描述子（occupy_array_）计算相似度；非0=使用语义描述子（语义标签序列）计算相似度
  float semantic_ratio_threshold_ = 0.5;    // 语义描述子相似度：两格点标签相同且比例差小于此阈值时计为匹配（范围0.0-1.0）
  
  /* for semantic-weighted ICP */
  float semantic_icp_weight_ = 2.0;        // 语义匹配点的ICP权重（相对于非匹配点）
                                             // 当语义标签匹配时，该点的权重会乘以这个值
                                             // 典型值：1.5-3.0，设置为1.0表示不使用语义权重

} ConfigSetting;

typedef struct BinaryDescriptor {
  std::vector<bool> occupy_array_;
  unsigned char summary_;
  Eigen::Vector3d location_;
  // 语义描述子：与 occupy_array_ 同序的语义标签序列，每格取该格内最多语义标签及其比例
  std::vector<uint32_t> semantic_label_array_;
  std::vector<double> semantic_ratio_array_;
  // 保留用于可视化或兼容，不再用于匹配
  uint32_t semantic_label_ = 0;
  double semantic_ratio_ = 0.0;
  uint32_t semantic_label_2_ = 0;
  double semantic_ratio_2_ = 0.0;
} BinaryDescriptor;

// Semantic Binary Triangle Descriptor
typedef struct SemanticTriangularDescriptor {
  Eigen::Vector3d triangle_;
  Eigen::Vector3d angle_;
  Eigen::Vector3d center_;
  unsigned short frame_number_;
  BinaryDescriptor binary_A_;
  BinaryDescriptor binary_B_;
  BinaryDescriptor binary_C_;
  // 三个顶点的语义标签（按边长顺序：A, B, C）- 第一标签
  Eigen::Vector3d vertex_semantic_;
  // 三个顶点的语义标签比重（按边长顺序：A, B, C，范围0-1）- 第一标签
  Eigen::Vector3d vertex_semantic_ratio_;
  // 三个顶点的语义标签（按边长顺序：A, B, C）- 第二标签
  Eigen::Vector3d vertex_semantic_2_;
  // 三个顶点的语义标签比重（按边长顺序：A, B, C，范围0-1）- 第二标签
  Eigen::Vector3d vertex_semantic_ratio_2_;
} SemanticTriangularDescriptor;

// 保持向后兼容的别名
typedef SemanticTriangularDescriptor BTC;

typedef struct Plane {
  pcl::PointXYZINormal p_center_;
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  Eigen::Matrix3d covariance_;
  float radius_ = 0;
  float min_eigen_value_ = 1;
  float d_ = 0;
  int id_ = 0;
  int sub_plane_num_ = 0;
  int points_size_ = 0;
  bool is_plane_ = false;
} Plane;

typedef struct SemanticTriangularMatchList {
  std::vector<std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor>> match_list_;
  std::pair<int, int> match_id_;
  int match_frame_;
  double mean_dis_;
} SemanticTriangularMatchList;

// 保持向后兼容的别名
typedef SemanticTriangularMatchList BTCMatchList;

struct M_POINT {
  float xyz[3];
  float intensity;
  int count = 0;
};

class VOXEL_LOC {
 public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

// Hash value
namespace std {
template <>
struct hash<VOXEL_LOC> {
  int64 operator()(const VOXEL_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
}  // namespace std

class SemanticTriangularDescriptor_LOC {
 public:
  int64_t x, y, z, a, b, c;

  SemanticTriangularDescriptor_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0, int64_t va = 0,
          int64_t vb = 0, int64_t vc = 0)
      : x(vx), y(vy), z(vz), a(va), b(vb), c(vc) {}

  bool operator==(const SemanticTriangularDescriptor_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
    // return (x == other.x && y == other.y && z == other.z && a == other.a &&
    //         b == other.b && c == other.c);
  }
};

namespace std {
template <>
struct hash<SemanticTriangularDescriptor_LOC> {
  int64 operator()(const SemanticTriangularDescriptor_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
}  // namespace std

// 保持向后兼容的别名
typedef SemanticTriangularDescriptor_LOC BTC_LOC;

class OctoTree {
 public:
  ConfigSetting config_setting_;
  std::vector<Eigen::Vector3d> voxel_points_;
  std::shared_ptr<Plane> plane_ptr_;
  int layer_;
  int octo_state_;  // 0 is end of tree, 1 is not
  int merge_num_ = 0;
  bool is_project_ = false;
  std::vector<Eigen::Vector3d> project_normal;
  bool is_publish_ = false;
  OctoTree *leaves_[8];
  double voxel_center_[3];  // x, y, z
  float quater_length_;
  bool init_octo_;

  // for plot
  bool is_check_connect_[6];
  bool connect_[6];
  OctoTree *connect_tree_[6];

  OctoTree(const ConfigSetting &config_setting)
      : config_setting_(config_setting) {
    voxel_points_.clear();
    octo_state_ = 0;
    layer_ = 0;
    init_octo_ = false;
    for (int i = 0; i < 8; i++) {
      leaves_[i] = nullptr;
    }
    // for plot
    for (int i = 0; i < 6; i++) {
      is_check_connect_[i] = false;
      connect_[i] = false;
      connect_tree_[i] = nullptr;
    }
    plane_ptr_.reset(new Plane);
  }
  void init_plane();
  void init_octo_tree();
};

void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                         double voxel_size);

void load_config_setting(std::string &config_file,
                         ConfigSetting &config_setting);

double binary_similarity(const BinaryDescriptor &b1,
                         const BinaryDescriptor &b2);

bool binary_greater_sort(BinaryDescriptor a, BinaryDescriptor b);
bool plane_greater_sort(std::shared_ptr<Plane> plane1,
                        std::shared_ptr<Plane> plane2);

void publish_std(const std::vector<std::pair<BTC, BTC>> &match_std_list,
                 const Eigen::Matrix4d &transform1,
                 const Eigen::Matrix4d &transform2,
                 const ros::Publisher &std_publisher);

void publish_std_list(const std::vector<BTC> &btc_list,
                      const ros::Publisher &std_publisher);

void publish_binary(const std::vector<BinaryDescriptor> &binary_list,
                    const Eigen::Vector3d &text_color,
                    const std::string &text_ns,
                    const ros::Publisher &text_publisher);

double calc_triangle_dis(
    const std::vector<std::pair<BTC, BTC>> &match_std_list);

double calc_binary_similaity(
    const std::vector<std::pair<BTC, BTC>> &match_std_list);

void CalcQuation(const Eigen::Vector3d &vec, const int axis,
                 geometry_msgs::Quaternion &q);

void pubPlane(const ros::Publisher &plane_pub, const std::string plane_ns,
              const int plane_id, const pcl::PointXYZINormal normal_p,
              const float radius, const Eigen::Vector3d rgb);

struct PlaneSolver {
  PlaneSolver(Eigen::Vector3d curr_point_, Eigen::Vector3d curr_normal_,
              Eigen::Vector3d target_point_, Eigen::Vector3d target_normal_)
      : curr_point(curr_point_),
        curr_normal(curr_normal_),
        target_point(target_point_),
        target_normal(target_normal_) {};
  template <typename T>
  bool operator()(const T *q, const T *t, T *residual) const {
    Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
    Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
    Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()),
                              T(curr_point.z())};
    Eigen::Matrix<T, 3, 1> point_w;
    point_w = q_w_curr * cp + t_w_curr;
    Eigen::Matrix<T, 3, 1> point_target(
        T(target_point.x()), T(target_point.y()), T(target_point.z()));
    Eigen::Matrix<T, 3, 1> norm(T(target_normal.x()), T(target_normal.y()),
                                T(target_normal.z()));
    residual[0] = norm.dot(point_w - point_target);
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_,
                                     const Eigen::Vector3d curr_normal_,
                                     Eigen::Vector3d target_point_,
                                     Eigen::Vector3d target_normal_) {
    return (
        new ceres::AutoDiffCostFunction<PlaneSolver, 1, 4, 3>(new PlaneSolver(
            curr_point_, curr_normal_, target_point_, target_normal_)));
  }

  Eigen::Vector3d curr_point;
  Eigen::Vector3d curr_normal;
  Eigen::Vector3d target_point;
  Eigen::Vector3d target_normal;
};

class SemanticTriangularDescManager {
 public:
  SemanticTriangularDescManager() = default;

  ConfigSetting config_setting_;

  SemanticTriangularDescManager(ConfigSetting &config_setting)
      : config_setting_(config_setting) {};

  // if print debug info
  bool print_debug_info_ = false;

  // hash table, save all descriptors
  std::unordered_map<SemanticTriangularDescriptor_LOC, std::vector<SemanticTriangularDescriptor>> data_base_;

  // save all key clouds, optional
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> key_cloud_vec_;

  // save all binary descriptors of key frame
  std::vector<std::vector<BinaryDescriptor>> history_binary_list_;

  // save all planes of key frame, required
  std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> plane_cloud_vec_;
  
  // save semantic voxel maps for each frame (for semantic-weighted ICP)
  std::vector<std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>>> semantic_voxel_map_vec_;

  /*Three main processing functions*/

  // generate SemanticTriangularDescriptors from a point cloud with labels
  void GenerateSemanticTriangularDescs(const pcl::PointCloud<pcl::PointXYZL>::Ptr &input_cloud,
                        const int frame_id, std::vector<SemanticTriangularDescriptor> &stds_vec);

  // generate BtcDescs from a point cloud (backward compatibility)
  void GenerateBtcDescs(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                        const int frame_id, std::vector<BTC> &btcs_vec);

  // search result <candidate_id, plane icp score>. -1 for no loop
  // Note: BTC is a typedef of SemanticTriangularDescriptor, so this handles both semantic and backward-compatible cases
  void SearchLoop(const std::vector<SemanticTriangularDescriptor> &stds_vec,
                  std::pair<int, double> &loop_result,
                  std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform,
                  std::vector<std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor>> &loop_std_pair);
  
  // Backward compatibility: same function due to type aliasing (BTC = SemanticTriangularDescriptor)
  // This is just for documentation - the implementation is the same as above

  // add descriptors to database
  // Note: BTC is a typedef of SemanticTriangularDescriptor, so this handles both cases
  void AddSemanticTriangularDescs(const std::vector<SemanticTriangularDescriptor> &stds_vec);
  
  // Backward compatibility alias (same as AddSemanticTriangularDescs due to type aliasing)
  void AddBtcDescs(const std::vector<BTC> &btcs_vec) {
    AddSemanticTriangularDescs(btcs_vec);
  }

  // Geometrical optimization by plane-to-plane icp
  // If source_semantic_map and target_semantic_map are provided, semantic-weighted ICP will be used
  void PlaneGeomrtricIcp(
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
      std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform,
      const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> *source_semantic_map = nullptr,
      const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> *target_semantic_map = nullptr);

 private:
  /*Following are sub-processing functions*/

  // voxelization and plane detection with semantic labels
  void init_voxel_map_semantic(const pcl::PointCloud<pcl::PointXYZL>::Ptr &input_cloud,
                      std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map);

  // voxelization and plane detection (backward compatibility)
  void init_voxel_map(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                      std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map);

  // acquire planes from voxel_map with semantic labels
  void get_plane_semantic(const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map,
                 pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud,
                 std::unordered_set<VOXEL_LOC> &plane_voxels);

  // acquire planes from voxel_map (backward compatibility)
  void get_plane(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                 pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud);

  void get_project_plane(
      std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
      std::vector<std::shared_ptr<Plane>> &project_plane_list);

  void merge_plane(std::vector<std::shared_ptr<Plane>> &origin_list,
                   std::vector<std::shared_ptr<Plane>> &merge_plane_list);

  // extract corner points from pre-build voxel map and clouds with semantic labels
  void binary_extractor_semantic(
      const std::vector<std::shared_ptr<Plane>> proj_plane_list,
      const pcl::PointCloud<pcl::PointXYZL>::Ptr &input_cloud,
      const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map,
      std::vector<BinaryDescriptor> &binary_descriptor_list);

  // extract corner points from pre-build voxel map and clouds (backward compatibility)
  void binary_extractor(
      const std::vector<std::shared_ptr<Plane>> proj_plane_list,
      const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
      std::vector<BinaryDescriptor> &binary_descriptor_list);

  void extract_binary(const Eigen::Vector3d &project_center,
                      const Eigen::Vector3d &project_normal,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                      std::vector<BinaryDescriptor> &binary_list);

  void extract_binary_semantic(const Eigen::Vector3d &project_center,
                      const Eigen::Vector3d &project_normal,
                      const pcl::PointCloud<pcl::PointXYZL>::Ptr &input_cloud,
                      const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map,
                      std::vector<BinaryDescriptor> &binary_list);

  // non maximum suppression, to control the number of corners
  void non_maxi_suppression(std::vector<BinaryDescriptor> &binary_list);

  // build SemanticTriangularDescriptors from corner points with semantic labels
  void generate_semantic_triangular_desc(const std::vector<BinaryDescriptor> &binary_list,
                    const int &frame_id, std::vector<SemanticTriangularDescriptor> &std_list);

  // build BTCs from corner points (backward compatibility)
  void generate_btc(const std::vector<BinaryDescriptor> &binary_list,
                    const int &frame_id, std::vector<BTC> &btc_list);

  // Select a specified number of candidate frames according to the number of
  // SemanticTriangularDescriptors rough matches
  void candidate_selector(const std::vector<SemanticTriangularDescriptor> &stds_vec,
                          std::vector<SemanticTriangularMatchList> &candidate_matcher_vec);

  // Get the best candidate frame by geometry check
  void candidate_verify(
      const SemanticTriangularMatchList &candidate_matcher, double &verify_score,
      std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
      std::vector<std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor>> &sucess_match_vec);

  // Get the transform between a matched std pair
  void triangle_solver(std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor> &std_pair, Eigen::Vector3d &t,
                       Eigen::Matrix3d &rot);

  // Geometrical verification by plane-to-plane icp threshold
  double plane_geometric_verify(
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
      const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform);

  // Helper function: get semantic label at location (most frequent label in voxel)
  uint32_t get_semantic_label_at_location(
      const Eigen::Vector3d &location,
      const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map);
  
  // Helper function: get semantic ratio from voxel map
  double get_semantic_ratio(const Eigen::Vector3d &location, uint32_t semantic_label,
                            const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map);
  
  // Helper function: get first and second semantic labels and ratios at location
  void get_semantic_labels_at_location(
      const Eigen::Vector3d &location,
      const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map,
      uint32_t &label_1, double &ratio_1, uint32_t &label_2, double &ratio_2);
};

// 保持向后兼容的别名
typedef SemanticTriangularDescManager BtcDescManager;

#endif  // SEMANTIC_TRIANGULAR_DESCRIPTOR_H
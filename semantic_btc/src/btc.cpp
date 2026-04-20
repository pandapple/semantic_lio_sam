#include "include/btc.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <unordered_set>

void load_config_setting(std::string &config_file,
                         ConfigSetting &config_setting) {
  cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
  if (!fSettings.isOpened()) {
    std::cerr << "Failed to open settings file at: " << config_file
              << std::endl;
    exit(-1);
  }

  // for binary descriptor
  config_setting.useful_corner_num_ = fSettings["useful_corner_num"];
  config_setting.plane_merge_normal_thre_ =
      fSettings["plane_merge_normal_thre"];
  config_setting.plane_merge_dis_thre_ = fSettings["plane_merge_dis_thre"];
  config_setting.plane_detection_thre_ = fSettings["plane_detection_thre"];
  config_setting.voxel_size_ = fSettings["voxel_size"];
  config_setting.voxel_init_num_ = fSettings["voxel_init_num"];
  config_setting.proj_plane_num_ = fSettings["proj_plane_num"];
  config_setting.proj_image_resolution_ = fSettings["proj_image_resolution"];
  config_setting.proj_image_high_inc_ = fSettings["proj_image_high_inc"];
  config_setting.proj_dis_min_ = fSettings["proj_dis_min"];
  config_setting.proj_dis_max_ = fSettings["proj_dis_max"];
  config_setting.summary_min_thre_ = fSettings["summary_min_thre"];
  config_setting.line_filter_enable_ = fSettings["line_filter_enable"];

  // std descriptor
  config_setting.descriptor_near_num_ = fSettings["descriptor_near_num"];
  config_setting.descriptor_min_len_ = fSettings["descriptor_min_len"];
  config_setting.descriptor_max_len_ = fSettings["descriptor_max_len"];
  // Support both "non_max_suppression_radius" and "max_constrait_dis" for backward compatibility
  if (fSettings["non_max_suppression_radius"].empty()) {
  config_setting.non_max_suppression_radius_ = fSettings["max_constrait_dis"];
  } else {
    config_setting.non_max_suppression_radius_ = fSettings["non_max_suppression_radius"];
  }
  config_setting.std_side_resolution_ = fSettings["triangle_resolution"];

  // candidate search
  config_setting.skip_near_num_ = fSettings["skip_near_num"];
  config_setting.candidate_num_ = fSettings["candidate_num"];
  config_setting.rough_dis_threshold_ = fSettings["rough_dis_threshold"];
  config_setting.similarity_threshold_ = fSettings["similarity_threshold"];
  config_setting.icp_threshold_ = fSettings["icp_threshold"];
  config_setting.normal_threshold_ = fSettings["normal_threshold"];
  config_setting.dis_threshold_ = fSettings["dis_threshold"];

  // semantic matching parameters
  if (!fSettings["semantic_vertex_match_threshold"].empty()) {
    config_setting.semantic_vertex_match_threshold_ = fSettings["semantic_vertex_match_threshold"];
  }
  if (!fSettings["semantic_ratio_threshold"].empty()) {
    config_setting.semantic_ratio_threshold_ = fSettings["semantic_ratio_threshold"];
  }
  if (!fSettings["semantic_icp_weight"].empty()) {
    config_setting.semantic_icp_weight_ = fSettings["semantic_icp_weight"];
  }

  std::cout << "Sucessfully load config file:" << config_file << std::endl;
}

void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                         double voxel_size) {
  int intensity = rand() % 255;
  if (voxel_size < 0.01) {
    return;
  }
  std::unordered_map<VOXEL_LOC, M_POINT> voxel_map;
  uint plsize = pl_feat.size();

  for (uint i = 0; i < plsize; i++) {
    pcl::PointXYZI &p_c = pl_feat[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      iter->second.xyz[0] += p_c.x;
      iter->second.xyz[1] += p_c.y;
      iter->second.xyz[2] += p_c.z;
      iter->second.intensity += p_c.intensity;
      iter->second.count++;
    } else {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.intensity = p_c.intensity;
      anp.count = 1;
      voxel_map[position] = anp;
    }
  }
  plsize = voxel_map.size();
  pl_feat.clear();
  pl_feat.resize(plsize);

  uint i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter) {
    pl_feat[i].x = iter->second.xyz[0] / iter->second.count;
    pl_feat[i].y = iter->second.xyz[1] / iter->second.count;
    pl_feat[i].z = iter->second.xyz[2] / iter->second.count;
    pl_feat[i].intensity = iter->second.intensity / iter->second.count;
    i++;
  }
}
double binary_similarity(const BinaryDescriptor &b1,
                         const BinaryDescriptor &b2) {
  double dis = 0;
  for (size_t i = 0; i < b1.occupy_array_.size(); i++) {
    if (b1.occupy_array_[i] == true && b2.occupy_array_[i] == true) {
      dis += 1;
    }
  }
  return 2 * dis / (b1.summary_ + b2.summary_);
}

// 语义描述子相似度：按序比较语义标签序列，标签相同且比例差 < semantic_ratio_threshold 计为匹配
double semantic_descriptor_similarity(const BinaryDescriptor &b1,
                                     const BinaryDescriptor &b2,
                                     float semantic_ratio_threshold) {
  if (b1.semantic_label_array_.empty() || b2.semantic_label_array_.size() != b1.semantic_label_array_.size()) {
    return 0.0;
  }
  double match = 0;
  for (size_t i = 0; i < b1.semantic_label_array_.size(); i++) {
    if (b1.semantic_label_array_[i] != b2.semantic_label_array_[i]) continue;
    double ratio_diff = std::abs(b1.semantic_ratio_array_[i] - b2.semantic_ratio_array_[i]);
    if (ratio_diff < semantic_ratio_threshold) {
      match += 1.0;
    }
  }
  size_t n = b1.semantic_label_array_.size();
  if (n == 0) return 0.0;
  return 2 * match / (n + n);
}

bool binary_greater_sort(BinaryDescriptor a, BinaryDescriptor b) {
  return (a.summary_ > b.summary_);
}

bool plane_greater_sort(std::shared_ptr<Plane> plane1,
                        std::shared_ptr<Plane> plane2) {
  return plane1->points_size_ > plane2->points_size_;
}

void OctoTree::init_octo_tree() {
  if (voxel_points_.size() > config_setting_.voxel_init_num_) {
    init_plane();
  }
}

void OctoTree::init_plane() {
  plane_ptr_->covariance_ = Eigen::Matrix3d::Zero();
  plane_ptr_->center_ = Eigen::Vector3d::Zero();
  plane_ptr_->normal_ = Eigen::Vector3d::Zero();
  plane_ptr_->points_size_ = voxel_points_.size();
  plane_ptr_->radius_ = 0;
  for (auto pi : voxel_points_) {
    plane_ptr_->covariance_ += pi * pi.transpose();
    plane_ptr_->center_ += pi;
  }
  plane_ptr_->center_ = plane_ptr_->center_ / plane_ptr_->points_size_;
  plane_ptr_->covariance_ =
      plane_ptr_->covariance_ / plane_ptr_->points_size_ -
      plane_ptr_->center_ * plane_ptr_->center_.transpose();
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane_ptr_->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  Eigen::Matrix3d::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  if (evalsReal(evalsMin) < config_setting_.plane_detection_thre_) {
    plane_ptr_->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
        evecs.real()(2, evalsMin);
    plane_ptr_->min_eigen_value_ = evalsReal(evalsMin);
    plane_ptr_->radius_ = sqrt(evalsReal(evalsMax));
    plane_ptr_->is_plane_ = true;

    plane_ptr_->d_ = -(plane_ptr_->normal_(0) * plane_ptr_->center_(0) +
                       plane_ptr_->normal_(1) * plane_ptr_->center_(1) +
                       plane_ptr_->normal_(2) * plane_ptr_->center_(2));
    plane_ptr_->p_center_.x = plane_ptr_->center_(0);
    plane_ptr_->p_center_.y = plane_ptr_->center_(1);
    plane_ptr_->p_center_.z = plane_ptr_->center_(2);
    plane_ptr_->p_center_.normal_x = plane_ptr_->normal_(0);
    plane_ptr_->p_center_.normal_y = plane_ptr_->normal_(1);
    plane_ptr_->p_center_.normal_z = plane_ptr_->normal_(2);
  } else {
    plane_ptr_->is_plane_ = false;
  }
}

void publish_binary(const std::vector<BinaryDescriptor> &binary_list,
                    const Eigen::Vector3d &text_color,
                    const std::string &text_ns,
                    const ros::Publisher &text_publisher) {
  visualization_msgs::MarkerArray text_array;
  visualization_msgs::Marker text;
  text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  text.action = visualization_msgs::Marker::ADD;
  text.ns = text_ns;
  text.color.a = 0.8;  // Don't forget to set the alpha!
  text.scale.z = 0.08;
  text.pose.orientation.w = 1.0;
  text.header.frame_id = "camera_init";
  for (size_t i = 0; i < binary_list.size(); i++) {
    text.pose.position.x = binary_list[i].location_[0];
    text.pose.position.y = binary_list[i].location_[1];
    text.pose.position.z = binary_list[i].location_[2];
    std::ostringstream str;
    str << std::to_string((int)(binary_list[i].summary_));
    text.text = str.str();
    text.scale.x = 0.5;
    text.scale.y = 0.5;
    text.scale.z = 0.5;
    text.color.r = text_color[0];
    text.color.g = text_color[1];
    text.color.b = text_color[2];
    text.color.a = 1;
    text.id++;
    text_array.markers.push_back(text);
  }
  for (int i = 1; i < 100; i++) {
    text.color.a = 0;
    text.id++;
    text_array.markers.push_back(text);
  }
  text_publisher.publish(text_array);
  return;
}

void publish_std_list(const std::vector<BTC> &btc_list,
                      const ros::Publisher &std_publisher) {
  // publish descriptor
  visualization_msgs::MarkerArray ma_line;
  visualization_msgs::Marker m_line;
  m_line.type = visualization_msgs::Marker::LINE_LIST;
  m_line.action = visualization_msgs::Marker::ADD;
  m_line.ns = "std";
  // Don't forget to set the alpha!
  m_line.scale.x = 0.5;
  m_line.pose.orientation.w = 1.0;
  m_line.header.frame_id = "camera_init";
  m_line.id = 0;
  m_line.points.clear();
  m_line.color.r = 0;
  m_line.color.g = 1;
  m_line.color.b = 0;
  m_line.color.a = 1;
  for (auto var : btc_list) {
    geometry_msgs::Point p;
    p.x = var.binary_A_.location_[0];
    p.y = var.binary_A_.location_[1];
    p.z = var.binary_A_.location_[2];
    m_line.points.push_back(p);
    p.x = var.binary_B_.location_[0];
    p.y = var.binary_B_.location_[1];
    p.z = var.binary_B_.location_[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    p.x = var.binary_C_.location_[0];
    p.y = var.binary_C_.location_[1];
    p.z = var.binary_C_.location_[2];
    m_line.points.push_back(p);
    p.x = var.binary_B_.location_[0];
    p.y = var.binary_B_.location_[1];
    p.z = var.binary_B_.location_[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    p.x = var.binary_C_.location_[0];
    p.y = var.binary_C_.location_[1];
    p.z = var.binary_C_.location_[2];
    m_line.points.push_back(p);
    p.x = var.binary_A_.location_[0];
    p.y = var.binary_A_.location_[1];
    p.z = var.binary_A_.location_[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
  }
  for (int j = 0; j < 1000 * 3; j++) {
    m_line.color.a = 0.00;
    ma_line.markers.push_back(m_line);
    m_line.id++;
  }
  std_publisher.publish(ma_line);
  m_line.id = 0;
  ma_line.markers.clear();
}

void publish_std(const std::vector<std::pair<BTC, BTC>> &match_std_list,
                 const Eigen::Matrix4d &transform1,
                 const Eigen::Matrix4d &transform2,
                 const ros::Publisher &std_publisher) {
  // publish descriptor
  // bool transform_enable = true;
  visualization_msgs::MarkerArray ma_line;
  visualization_msgs::Marker m_line;
  m_line.type = visualization_msgs::Marker::LINE_LIST;
  m_line.action = visualization_msgs::Marker::ADD;
  m_line.ns = "lines";
  // Don't forget to set the alpha!
  m_line.scale.x = 0.25;
  m_line.pose.orientation.w = 1.0;
  m_line.header.frame_id = "camera_init";
  m_line.id = 0;
  int max_pub_cnt = 1;
  for (auto var : match_std_list) {
    if (max_pub_cnt > 100) {
      break;
    }
    max_pub_cnt++;
    m_line.color.a = 0.8;
    m_line.points.clear();
    // m_line.color.r = 0 / 255;
    // m_line.color.g = 233.0 / 255;
    // m_line.color.b = 0 / 255;
    m_line.color.r = 252.0 / 255;
    m_line.color.g = 233.0 / 255;
    m_line.color.b = 79.0 / 255;
    geometry_msgs::Point p;
    Eigen::Vector3d t_p;
    t_p = var.second.binary_A_.location_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);

    t_p = var.second.binary_B_.location_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();

    t_p = var.second.binary_C_.location_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);

    t_p = var.second.binary_B_.location_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();

    t_p = var.second.binary_C_.location_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);

    t_p = var.second.binary_A_.location_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    // another
    m_line.points.clear();
    // 252; 233; 79

    m_line.color.r = 1;
    m_line.color.g = 1;
    m_line.color.b = 1;
    // m_line.color.r = 252.0 / 255;
    // m_line.color.g = 233.0 / 255;
    // m_line.color.b = 79.0 / 255;
    t_p = var.first.binary_A_.location_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);

    t_p = var.first.binary_B_.location_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();

    t_p = var.first.binary_C_.location_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    t_p = var.first.binary_B_.location_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();

    t_p = var.first.binary_C_.location_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    t_p = var.first.binary_A_.location_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    // debug
    // std_publisher.publish(ma_line);
    // std::cout << "var first: " << var.first.triangle_.transpose()
    //           << " , var second: " << var.second.triangle_.transpose()
    //           << std::endl;
    // getchar();
  }
  for (int j = 0; j < 100 * 6; j++) {
    m_line.color.a = 0.00;
    ma_line.markers.push_back(m_line);
    m_line.id++;
  }
  std_publisher.publish(ma_line);
  m_line.id = 0;
  ma_line.markers.clear();
}

double calc_triangle_dis(
    const std::vector<std::pair<BTC, BTC>> &match_std_list) {
  double mean_triangle_dis = 0;
  for (auto var : match_std_list) {
    mean_triangle_dis += (var.first.triangle_ - var.second.triangle_).norm() /
                         var.first.triangle_.norm();
  }
  if (match_std_list.size() > 0) {
    mean_triangle_dis = mean_triangle_dis / match_std_list.size();
  } else {
    mean_triangle_dis = -1;
  }
  return mean_triangle_dis;
}

double calc_binary_similaity(
    const std::vector<std::pair<BTC, BTC>> &match_std_list) {
  double mean_binary_similarity = 0;
  for (auto var : match_std_list) {
    mean_binary_similarity +=
        (binary_similarity(var.first.binary_A_, var.second.binary_A_) +
         binary_similarity(var.first.binary_B_, var.second.binary_B_) +
         binary_similarity(var.first.binary_C_, var.second.binary_C_)) /
        3;
  }
  if (match_std_list.size() > 0) {
    mean_binary_similarity = mean_binary_similarity / match_std_list.size();
  } else {
    mean_binary_similarity = -1;
  }
  return mean_binary_similarity;
}

void CalcQuation(const Eigen::Vector3d &vec, const int axis,
                 geometry_msgs::Quaternion &q) {
  Eigen::Vector3d x_body = vec;
  Eigen::Vector3d y_body(1, 1, 0);
  if (x_body(2) != 0) {
    y_body(2) = -(y_body(0) * x_body(0) + y_body(1) * x_body(1)) / x_body(2);
  } else {
    if (x_body(1) != 0) {
      y_body(1) = -(y_body(0) * x_body(0)) / x_body(1);
    } else {
      y_body(0) = 0;
    }
  }
  y_body.normalize();
  Eigen::Vector3d z_body = x_body.cross(y_body);
  Eigen::Matrix3d rot;

  rot << x_body(0), x_body(1), x_body(2), y_body(0), y_body(1), y_body(2),
      z_body(0), z_body(1), z_body(2);
  Eigen::Matrix3d rotation = rot.transpose();
  if (axis == 2) {
    Eigen::Matrix3d rot_inc;
    rot_inc << 0, 0, 1, 0, 1, 0, -1, 0, 0;
    rotation = rotation * rot_inc;
  }
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void pubPlane(const ros::Publisher &plane_pub, const std::string plane_ns,
              const int plane_id, const pcl::PointXYZINormal normal_p,
              const float radius, const Eigen::Vector3d rgb) {
  visualization_msgs::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = ros::Time();
  plane.ns = plane_ns;
  plane.id = plane_id;
  plane.type = visualization_msgs::Marker::CUBE;
  plane.action = visualization_msgs::Marker::ADD;
  plane.pose.position.x = normal_p.x;
  plane.pose.position.y = normal_p.y;
  plane.pose.position.z = normal_p.z;
  geometry_msgs::Quaternion q;
  Eigen::Vector3d normal_vec(normal_p.normal_x, normal_p.normal_y,
                             normal_p.normal_z);
  CalcQuation(normal_vec, 2, q);
  plane.pose.orientation = q;
  plane.scale.x = 3.0 * radius;
  plane.scale.y = 3.0 * radius;
  plane.scale.z = 0.1;
  plane.color.a = 0.8;  // 0.8
  plane.color.r = fabs(rgb(0));
  plane.color.g = fabs(rgb(1));
  plane.color.b = fabs(rgb(2));
  plane.lifetime = ros::Duration();
  plane_pub.publish(plane);
}

void SemanticTriangularDescManager::GenerateSemanticTriangularDescs(
    const pcl::PointCloud<pcl::PointXYZL>::Ptr &input_cloud, const int frame_id,
    std::vector<SemanticTriangularDescriptor> &stds_vec) {  // step1, voxelization with semantic labels, then build OctoTree for plane detection (same as original)
  // First, create semantic voxel map to preserve label information
  std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> semantic_voxel_map;
  init_voxel_map_semantic(input_cloud, semantic_voxel_map);
  
  // Then, build OctoTree structure from semantic voxel map (same as original version)
  std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
  for (auto iter = semantic_voxel_map.begin(); iter != semantic_voxel_map.end(); iter++) {
    if (iter->second.size() < config_setting_.voxel_init_num_) {
      continue;
    }
    OctoTree *octo_tree = new OctoTree(config_setting_);
    for (const auto& point_label : iter->second) {
      octo_tree->voxel_points_.push_back(point_label.first);
    }
    voxel_map[iter->first] = octo_tree;
  }
  
  // Initialize octo trees in parallel (same as original version)
  std::vector<std::unordered_map<VOXEL_LOC, OctoTree *>::iterator> iter_list;
  std::vector<size_t> index;
  size_t i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter) {
    index.push_back(i);
    i++;
    iter_list.push_back(iter);
  }
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) { iter_list[i]->second->init_octo_tree(); });
  
  // step2, get plane cloud (same as original version)
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  get_plane(voxel_map, plane_cloud);
  if (print_debug_info_) {
    std::cout << "[Description] planes size:" << plane_cloud->size()
              << std::endl;
  }

  plane_cloud_vec_.push_back(plane_cloud);
  
  // Save semantic voxel map for semantic-weighted ICP
  semantic_voxel_map_vec_.push_back(semantic_voxel_map);

  // step3, extraction binary descriptors with semantic labels
  // Use get_project_plane (same as original version) to get projection planes
  std::vector<std::shared_ptr<Plane>> proj_plane_list;
  std::vector<std::shared_ptr<Plane>> merge_plane_list;
  get_project_plane(voxel_map, proj_plane_list);
  if (proj_plane_list.size() == 0) {
    std::shared_ptr<Plane> single_plane(new Plane);
    single_plane->normal_ << 0, 0, 1;
    if (input_cloud->size() > 0) {
      single_plane->center_ << input_cloud->points[0].x, input_cloud->points[0].y,
          input_cloud->points[0].z;
    }
    merge_plane_list.push_back(single_plane);
  } else {
    sort(proj_plane_list.begin(), proj_plane_list.end(), plane_greater_sort);
    merge_plane(proj_plane_list, merge_plane_list);
    sort(merge_plane_list.begin(), merge_plane_list.end(), plane_greater_sort);
  }
  std::vector<BinaryDescriptor> binary_list;
  binary_extractor_semantic(merge_plane_list, input_cloud, semantic_voxel_map, binary_list);
  history_binary_list_.push_back(binary_list);
  if (print_debug_info_) {
    std::cout << "[Description] binary size:" << binary_list.size()
              << std::endl;
  }
  
  // 检查二进制描述子是否为空
  if (binary_list.empty()) {
    if (print_debug_info_) {
      std::cerr << "[GenerateSemanticTriangularDescs] Warning: No binary descriptors extracted! "
                << "plane_cloud size: " << plane_cloud->size() 
                << ", merge_plane_list size: " << merge_plane_list.size()
                << ", input_cloud size: " << input_cloud->size() << std::endl;
    }
    stds_vec.clear();
    // Clean up OctoTree memory
    for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
      delete (iter->second);
    }
    return;
  }

  // step4, generate stable semantic triangle descriptors
  stds_vec.clear();
  generate_semantic_triangular_desc(binary_list, frame_id, stds_vec);
  if (print_debug_info_) {
    std::cout << "[Description] semantic triangular descriptors size:" << stds_vec.size() << std::endl;
  }
  
  // step5, clear memory (same as original version)
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    delete (iter->second);
  }
  return;
}

void BtcDescManager::GenerateBtcDescs(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud, const int frame_id,
    std::vector<BTC> &btcs_vec) {  // step1, voxelization and plane dection
  std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
  init_voxel_map(input_cloud, voxel_map);
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  get_plane(voxel_map, plane_cloud);
  if (print_debug_info_) {
    std::cout << "[Description] planes size:" << plane_cloud->size()
              << std::endl;
  }

  plane_cloud_vec_.push_back(plane_cloud);

  // step3, extraction binary descriptors
  std::vector<std::shared_ptr<Plane>> proj_plane_list;
  std::vector<std::shared_ptr<Plane>> merge_plane_list;
  get_project_plane(voxel_map, proj_plane_list);
  if (proj_plane_list.size() == 0) {
    std::shared_ptr<Plane> single_plane(new Plane);
    single_plane->normal_ << 0, 0, 1;
    single_plane->center_ << input_cloud->points[0].x, input_cloud->points[0].y,
        input_cloud->points[0].z;
    merge_plane_list.push_back(single_plane);
  } else {
    sort(proj_plane_list.begin(), proj_plane_list.end(), plane_greater_sort);
    merge_plane(proj_plane_list, merge_plane_list);
    sort(merge_plane_list.begin(), merge_plane_list.end(), plane_greater_sort);
  }
  std::vector<BinaryDescriptor> binary_list;
  binary_extractor(merge_plane_list, input_cloud, binary_list);
  history_binary_list_.push_back(binary_list);
  // corner_cloud_vec_.push_back(corner_points);
  if (print_debug_info_) {
    std::cout << "[Description] binary size:" << binary_list.size()
              << std::endl;
  }

  // step4, generate stable triangle descriptors
  btcs_vec.clear();
  generate_btc(binary_list, frame_id, btcs_vec);
  if (print_debug_info_) {
    std::cout << "[Description] btcs size:" << btcs_vec.size() << std::endl;
  }
  // step5, clear memory
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    delete (iter->second);
  }
  return;
}

void SemanticTriangularDescManager::SearchLoop(
    const std::vector<SemanticTriangularDescriptor> &stds_vec, std::pair<int, double> &loop_result,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform,
    std::vector<std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor>> &loop_std_pair) {
  if (stds_vec.size() == 0) {
    if (print_debug_info_) {
      std::cerr << "No SemanticTriangularDescriptors!" << std::endl;
    }
    loop_result = std::pair<int, double>(-1, 0);
    return;
  }
  // step1, select candidates, default number 50
  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<SemanticTriangularMatchList> candidate_matcher_vec;
  candidate_selector(stds_vec, candidate_matcher_vec);
  
  if (print_debug_info_) {
    std::cout << "[SearchLoop] Debug: Found " << candidate_matcher_vec.size() 
              << " candidate matches after candidate_selector." << std::endl;
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  // step2, select best candidates from rough candidates
  double best_score = 0;
  int best_candidate_id = -1;
  int triggle_candidate = -1;
  std::pair<Eigen::Vector3d, Eigen::Matrix3d> best_transform;
  std::vector<std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor>> best_sucess_match_vec;
  for (size_t i = 0; i < candidate_matcher_vec.size(); i++) {
    double verify_score = -1;
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> relative_pose;
    std::vector<std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor>> sucess_match_vec;
    candidate_verify(candidate_matcher_vec[i], verify_score, relative_pose,
                     sucess_match_vec);
    if (print_debug_info_) {
      std::cout << "[Retreival] try frame:"
                << candidate_matcher_vec[i].match_id_.second << ", rough size:"
                << candidate_matcher_vec[i].match_list_.size()
                << ", score:" << verify_score << std::endl;
    }

    if (verify_score > best_score) {
      best_score = verify_score;
      best_candidate_id = candidate_matcher_vec[i].match_id_.second;
      best_transform = relative_pose;
      best_sucess_match_vec = sucess_match_vec;
      triggle_candidate = i;
    }
  }
  auto t3 = std::chrono::high_resolution_clock::now();

  if (print_debug_info_) {
    std::cout << "[Retreival] best candidate:" << best_candidate_id
              << ", score:" << best_score 
              << ", icp_threshold:" << config_setting_.icp_threshold_ << std::endl;
  }

  if (best_score > config_setting_.icp_threshold_) {
    loop_result = std::pair<int, double>(best_candidate_id, best_score);
    loop_transform = best_transform;
    loop_std_pair = best_sucess_match_vec;
    return;
  } else {
    // 即使低于阈值，也返回实际得分（用于评估）
    // 如果best_candidate_id >= 0，说明找到了候选，只是得分不够高
    if (best_candidate_id >= 0 && best_score > 0) {
      loop_result = std::pair<int, double>(best_candidate_id, best_score);
      loop_transform = best_transform;
      loop_std_pair = best_sucess_match_vec;
      if (print_debug_info_) {
        std::cerr << "[SearchLoop] Warning: Best score " << best_score 
                  << " is below threshold " << config_setting_.icp_threshold_ 
                  << ", but returning score for evaluation." << std::endl;
      }
    } else {
      // 没有找到任何候选或得分为0/负数
      if (print_debug_info_) {
        std::cerr << "[SearchLoop] Warning: No candidate found or all candidates failed verification. "
                  << "best_candidate_id=" << best_candidate_id 
                  << ", best_score=" << best_score << std::endl;
      }
    loop_result = std::pair<int, double>(-1, 0);
      if (print_debug_info_) {
        std::cerr << "[SearchLoop] Warning: No candidate found or all candidates failed verification." << std::endl;
      }
    }
    return;
  }
}

// Note: BtcDescManager::SearchLoop with BTC type is the same as SemanticTriangularDescManager::SearchLoop
// since BTC is a typedef of SemanticTriangularDescriptor. The implementation above handles both cases.

void SemanticTriangularDescManager::AddSemanticTriangularDescs(const std::vector<SemanticTriangularDescriptor> &stds_vec) {
  // update frame id
  for (auto single_std : stds_vec) {
    // calculate the position of single std
    SemanticTriangularDescriptor_LOC position;
    position.x = (int)(single_std.triangle_[0] + 0.5);
    position.y = (int)(single_std.triangle_[1] + 0.5);
    position.z = (int)(single_std.triangle_[2] + 0.5);
    auto iter = data_base_.find(position);
    if (iter != data_base_.end()) {
      data_base_[position].push_back(single_std);
    } else {
      std::vector<SemanticTriangularDescriptor> descriptor_vec;
      descriptor_vec.push_back(single_std);
      data_base_[position] = descriptor_vec;
    }
  }
  return;
}

// Note: BtcDescManager::AddBtcDescs with BTC type is the same as SemanticTriangularDescManager::AddSemanticTriangularDescs
// since BTC is a typedef of SemanticTriangularDescriptor. The implementation above handles both cases.

void BtcDescManager::PlaneGeomrtricIcp(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform,
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> *source_semantic_map,
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> *target_semantic_map) {
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  // 检查目标点云是否为空
  if (target_cloud->empty()) {
    if (print_debug_info_) {
      std::cerr << "[PlaneGeomrtricIcp] Warning: target_cloud is empty!" << std::endl;
    }
    return;
  }
  
  for (size_t i = 0; i < target_cloud->size(); i++) {
    pcl::PointXYZ pi;
    pi.x = target_cloud->points[i].x;
    pi.y = target_cloud->points[i].y;
    pi.z = target_cloud->points[i].z;
    input_cloud->push_back(pi);
  }
  kd_tree->setInputCloud(input_cloud);
  ceres::Manifold *quaternion_manifold = new ceres::EigenQuaternionManifold;
  ceres::Problem problem;
  ceres::LossFunction *loss_function = nullptr;
  Eigen::Matrix3d rot = transform.second;
  Eigen::Quaterniond q(rot);
  Eigen::Vector3d t = transform.first;
  double para_q[4] = {q.x(), q.y(), q.z(), q.w()};
  double para_t[3] = {t(0), t(1), t(2)};
  problem.AddParameterBlock(para_q, 4, quaternion_manifold);
  problem.AddParameterBlock(para_t, 3);
  Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
  Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  int useful_match = 0;
  int semantic_match_count = 0;
  
  // 检查是否启用语义权重ICP
  // 只有当 semantic_vertex_match_threshold_ > 0 时才使用语义权重
  // 如果阈值为0，则完全禁用语义功能（向后兼容模式）
  bool use_semantic_weight = (source_semantic_map != nullptr && 
                              target_semantic_map != nullptr && 
                              config_setting_.semantic_icp_weight_ > 1.0 &&
                              config_setting_.semantic_vertex_match_threshold_ > 0);
  
  for (size_t i = 0; i < source_cloud->size(); i++) {
    pcl::PointXYZINormal searchPoint = source_cloud->points[i];
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    pcl::PointXYZ use_search_point;
    use_search_point.x = pi[0];
    use_search_point.y = pi[1];
    use_search_point.z = pi[2];
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y,
                       searchPoint.normal_z);
    ni = rot * ni;
    if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      pcl::PointXYZINormal nearstPoint =
          target_cloud->points[pointIdxNKNSearch[0]];
      Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
      Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y,
                          nearstPoint.normal_z);
      Eigen::Vector3d normal_inc = ni - tni;
      Eigen::Vector3d normal_add = ni + tni;
      double point_to_point_dis = (pi - tpi).norm();
      double point_to_plane = fabs(tni.transpose() * (pi - tpi));
      if ((normal_inc.norm() < config_setting_.normal_threshold_ ||
           normal_add.norm() < config_setting_.normal_threshold_) &&
          point_to_plane < config_setting_.dis_threshold_ &&
          point_to_point_dis < 3) {
        useful_match++;
        
        // 检查语义标签是否匹配（与candidate_selector的逻辑保持一致）
        double weight = 1.0;
        if (use_semantic_weight) {
          // 获取源点和目标点的语义标签（第一和第二标签及其比例）
          Eigen::Vector3d source_point(searchPoint.x, searchPoint.y, searchPoint.z);
          Eigen::Vector3d target_point(nearstPoint.x, nearstPoint.y, nearstPoint.z);
          
          uint32_t src_label_1, src_label_2, tgt_label_1, tgt_label_2;
          double src_ratio_1, src_ratio_2, tgt_ratio_1, tgt_ratio_2;
          
          this->get_semantic_labels_at_location(source_point, *source_semantic_map, 
                                                 src_label_1, src_ratio_1, src_label_2, src_ratio_2);
          this->get_semantic_labels_at_location(target_point, *target_semantic_map,
                                                 tgt_label_1, tgt_ratio_1, tgt_label_2, tgt_ratio_2);
          
          // 检查4种可能的匹配组合（与candidate_selector的逻辑一致）
          bool semantic_match = false;
          
          // 1. src第一标签 == tgt第一标签
          if (src_label_1 != 0 && tgt_label_1 != 0 && src_label_1 == tgt_label_1) {
            double ratio_diff = std::abs(src_ratio_1 - tgt_ratio_1);
            if (ratio_diff < config_setting_.semantic_ratio_threshold_) {
              semantic_match = true;
            }
          }
          // 2. src第一标签 == tgt第二标签
          if (!semantic_match && src_label_1 != 0 && tgt_label_2 != 0 && src_label_1 == tgt_label_2) {
            double ratio_diff = std::abs(src_ratio_1 - tgt_ratio_2);
            if (ratio_diff < config_setting_.semantic_ratio_threshold_) {
              semantic_match = true;
            }
          }
          // 3. src第二标签 == tgt第一标签
          if (!semantic_match && src_label_2 != 0 && tgt_label_1 != 0 && src_label_2 == tgt_label_1) {
            double ratio_diff = std::abs(src_ratio_2 - tgt_ratio_1);
            if (ratio_diff < config_setting_.semantic_ratio_threshold_) {
              semantic_match = true;
            }
          }
          // 4. src第二标签 == tgt第二标签
          if (!semantic_match && src_label_2 != 0 && tgt_label_2 != 0 && src_label_2 == tgt_label_2) {
            double ratio_diff = std::abs(src_ratio_2 - tgt_ratio_2);
            if (ratio_diff < config_setting_.semantic_ratio_threshold_) {
              semantic_match = true;
            }
          }
          
          // 如果语义标签匹配，增加权重
          if (semantic_match) {
            weight = config_setting_.semantic_icp_weight_;
            semantic_match_count++;
          }
        }
        
        ceres::CostFunction *cost_function;
        Eigen::Vector3d curr_point(source_cloud->points[i].x,
                                   source_cloud->points[i].y,
                                   source_cloud->points[i].z);
        Eigen::Vector3d curr_normal(source_cloud->points[i].normal_x,
                                    source_cloud->points[i].normal_y,
                                    source_cloud->points[i].normal_z);

        cost_function = PlaneSolver::Create(curr_point, curr_normal, tpi, tni);
        
        // 使用 ScaledLoss 来应用权重
        ceres::LossFunction *weighted_loss = nullptr;
        if (weight > 1.0) {
          // 使用 ScaledLoss 来增加权重（权重越大，残差的影响越大）
          weighted_loss = new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP);
        }
        
        problem.AddResidualBlock(cost_function, weighted_loss, para_q, para_t);
      }
    }
  }
  
  if (print_debug_info_ && use_semantic_weight) {
    std::cout << "[PlaneGeomrtricIcp] Semantic-weighted ICP: " 
              << semantic_match_count << " / " << useful_match 
              << " matches have matching semantic labels" << std::endl;
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  Eigen::Quaterniond q_opt(para_q[3], para_q[0], para_q[1], para_q[2]);
  rot = q_opt.toRotationMatrix();
  t << t_last_curr(0), t_last_curr(1), t_last_curr(2);
  transform.first = t;
  transform.second = rot;
}

void BtcDescManager::init_voxel_map(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map) {
  uint plsize = input_cloud->size();
  for (uint i = 0; i < plsize; i++) {
    Eigen::Vector3d p_c(input_cloud->points[i].x, input_cloud->points[i].y,
                        input_cloud->points[i].z);
    double loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / config_setting_.voxel_size_;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      voxel_map[position]->voxel_points_.push_back(p_c);
    } else {
      OctoTree *octo_tree = new OctoTree(config_setting_);
      voxel_map[position] = octo_tree;
      voxel_map[position]->voxel_points_.push_back(p_c);
    }
  }
  std::vector<std::unordered_map<VOXEL_LOC, OctoTree *>::iterator> iter_list;
  std::vector<size_t> index;
  size_t i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter) {
    index.push_back(i);
    i++;
    iter_list.push_back(iter);
    // iter->second->init_octo_tree();
  }
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) { iter_list[i]->second->init_octo_tree(); });
}

void BtcDescManager::get_plane(
    const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud) {
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_plane_) {
      pcl::PointXYZINormal pi;
      pi.x = iter->second->plane_ptr_->center_[0];
      pi.y = iter->second->plane_ptr_->center_[1];
      pi.z = iter->second->plane_ptr_->center_[2];
      pi.normal_x = iter->second->plane_ptr_->normal_[0];
      pi.normal_y = iter->second->plane_ptr_->normal_[1];
      pi.normal_z = iter->second->plane_ptr_->normal_[2];
      plane_cloud->push_back(pi);
    }
  }
}

void SemanticTriangularDescManager::init_voxel_map_semantic(
    const pcl::PointCloud<pcl::PointXYZL>::Ptr &input_cloud,
    std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map) {
  uint plsize = input_cloud->size();
  for (uint i = 0; i < plsize; i++) {
    Eigen::Vector3d p_c(input_cloud->points[i].x, input_cloud->points[i].y,
                        input_cloud->points[i].z);
    uint32_t label = input_cloud->points[i].label & 0xFFFF;
    double loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / config_setting_.voxel_size_;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    voxel_map[position].push_back(std::make_pair(p_c, label));
  }
}

void SemanticTriangularDescManager::get_plane_semantic(
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud,
    std::unordered_set<VOXEL_LOC> &plane_voxels) {
  plane_voxels.clear();
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (iter->second.size() < config_setting_.voxel_init_num_) {
      continue;
    }
    
    // 使用PCA进行平面检测
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    
    for (const auto& point_label : iter->second) {
      const auto& point = point_label.first;
      covariance += point * point.transpose();
      center += point;
    }
    
    size_t point_count = iter->second.size();
    center /= point_count;
    covariance = covariance / point_count - center * center.transpose();
    
    // 特征值分解
    Eigen::EigenSolver<Eigen::Matrix3d> es(covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal = evals.real();
    
    // 找到最小特征值的索引
    Eigen::Matrix3d::Index evalsMin;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    
    // 如果最小特征值小于阈值，认为是平面
    if (evalsReal(evalsMin) < config_setting_.plane_detection_thre_) {
      pcl::PointXYZINormal pi;
      pi.x = center[0];
      pi.y = center[1];
      pi.z = center[2];
      pi.normal_x = evecs.real()(0, evalsMin);
      pi.normal_y = evecs.real()(1, evalsMin);
      pi.normal_z = evecs.real()(2, evalsMin);
      plane_cloud->push_back(pi);
      // 标记为平面体素
      plane_voxels.insert(iter->first);
    }
  }
}

void BtcDescManager::get_project_plane(
    std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
    std::vector<std::shared_ptr<Plane>> &project_plane_list) {
  std::vector<std::shared_ptr<Plane>> origin_list;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_plane_) {
      origin_list.push_back(iter->second->plane_ptr_);
    }
  }
  for (size_t i = 0; i < origin_list.size(); i++) origin_list[i]->id_ = 0;
  int current_id = 1;
  for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--) {
    for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++) {
      Eigen::Vector3d normal_diff = (*iter)->normal_ - (*iter2)->normal_;
      Eigen::Vector3d normal_add = (*iter)->normal_ + (*iter2)->normal_;
      double dis1 =
          fabs((*iter)->normal_(0) * (*iter2)->center_(0) +
               (*iter)->normal_(1) * (*iter2)->center_(1) +
               (*iter)->normal_(2) * (*iter2)->center_(2) + (*iter)->d_);
      double dis2 =
          fabs((*iter2)->normal_(0) * (*iter)->center_(0) +
               (*iter2)->normal_(1) * (*iter)->center_(1) +
               (*iter2)->normal_(2) * (*iter)->center_(2) + (*iter2)->d_);
      if (normal_diff.norm() < config_setting_.plane_merge_normal_thre_ ||
          normal_add.norm() < config_setting_.plane_merge_normal_thre_)
        if (dis1 < config_setting_.plane_merge_dis_thre_ &&
            dis2 < config_setting_.plane_merge_dis_thre_) {
          if ((*iter)->id_ == 0 && (*iter2)->id_ == 0) {
            (*iter)->id_ = current_id;
            (*iter2)->id_ = current_id;
            current_id++;
          } else if ((*iter)->id_ == 0 && (*iter2)->id_ != 0)
            (*iter)->id_ = (*iter2)->id_;
          else if ((*iter)->id_ != 0 && (*iter2)->id_ == 0)
            (*iter2)->id_ = (*iter)->id_;
        }
    }
  }
  std::vector<std::shared_ptr<Plane>> merge_list;
  std::vector<int> merge_flag;

  for (size_t i = 0; i < origin_list.size(); i++) {
    auto it =
        std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id_);
    if (it != merge_flag.end()) continue;
    if (origin_list[i]->id_ == 0) {
      continue;
    }
    std::shared_ptr<Plane> merge_plane(new Plane);
    (*merge_plane) = (*origin_list[i]);
    bool is_merge = false;
    for (size_t j = 0; j < origin_list.size(); j++) {
      if (i == j) continue;
      if (origin_list[j]->id_ == origin_list[i]->id_) {
        is_merge = true;
        Eigen::Matrix3d P_PT1 =
            (merge_plane->covariance_ +
             merge_plane->center_ * merge_plane->center_.transpose()) *
            merge_plane->points_size_;
        Eigen::Matrix3d P_PT2 =
            (origin_list[j]->covariance_ +
             origin_list[j]->center_ * origin_list[j]->center_.transpose()) *
            origin_list[j]->points_size_;
        Eigen::Vector3d merge_center =
            (merge_plane->center_ * merge_plane->points_size_ +
             origin_list[j]->center_ * origin_list[j]->points_size_) /
            (merge_plane->points_size_ + origin_list[j]->points_size_);
        Eigen::Matrix3d merge_covariance =
            (P_PT1 + P_PT2) /
                (merge_plane->points_size_ + origin_list[j]->points_size_) -
            merge_center * merge_center.transpose();
        merge_plane->covariance_ = merge_covariance;
        merge_plane->center_ = merge_center;
        merge_plane->points_size_ =
            merge_plane->points_size_ + origin_list[j]->points_size_;
        merge_plane->sub_plane_num_++;
        // for (size_t k = 0; k < origin_list[j]->cloud.size(); k++) {
        //   merge_plane->cloud.points.push_back(origin_list[j]->cloud.points[k]);
        // }
        Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance_);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3f::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        merge_plane->normal_ << evecs.real()(0, evalsMin),
            evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
        merge_plane->radius_ = sqrt(evalsReal(evalsMax));
        merge_plane->d_ = -(merge_plane->normal_(0) * merge_plane->center_(0) +
                            merge_plane->normal_(1) * merge_plane->center_(1) +
                            merge_plane->normal_(2) * merge_plane->center_(2));
        merge_plane->p_center_.x = merge_plane->center_(0);
        merge_plane->p_center_.y = merge_plane->center_(1);
        merge_plane->p_center_.z = merge_plane->center_(2);
        merge_plane->p_center_.normal_x = merge_plane->normal_(0);
        merge_plane->p_center_.normal_y = merge_plane->normal_(1);
        merge_plane->p_center_.normal_z = merge_plane->normal_(2);
      }
    }
    if (is_merge) {
      merge_flag.push_back(merge_plane->id_);
      merge_list.push_back(merge_plane);
    }
  }
  project_plane_list = merge_list;
}

void BtcDescManager::merge_plane(
    std::vector<std::shared_ptr<Plane>> &origin_list,
    std::vector<std::shared_ptr<Plane>> &merge_plane_list) {
  if (origin_list.size() == 1) {
    merge_plane_list = origin_list;
    return;
  }
  for (size_t i = 0; i < origin_list.size(); i++) origin_list[i]->id_ = 0;
  int current_id = 1;
  for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--) {
    for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++) {
      Eigen::Vector3d normal_diff = (*iter)->normal_ - (*iter2)->normal_;
      Eigen::Vector3d normal_add = (*iter)->normal_ + (*iter2)->normal_;
      double dis1 =
          fabs((*iter)->normal_(0) * (*iter2)->center_(0) +
               (*iter)->normal_(1) * (*iter2)->center_(1) +
               (*iter)->normal_(2) * (*iter2)->center_(2) + (*iter)->d_);
      double dis2 =
          fabs((*iter2)->normal_(0) * (*iter)->center_(0) +
               (*iter2)->normal_(1) * (*iter)->center_(1) +
               (*iter2)->normal_(2) * (*iter)->center_(2) + (*iter2)->d_);
      if (normal_diff.norm() < config_setting_.plane_merge_normal_thre_ ||
          normal_add.norm() < config_setting_.plane_merge_normal_thre_)
        if (dis1 < config_setting_.plane_merge_dis_thre_ &&
            dis2 < config_setting_.plane_merge_dis_thre_) {
          if ((*iter)->id_ == 0 && (*iter2)->id_ == 0) {
            (*iter)->id_ = current_id;
            (*iter2)->id_ = current_id;
            current_id++;
          } else if ((*iter)->id_ == 0 && (*iter2)->id_ != 0)
            (*iter)->id_ = (*iter2)->id_;
          else if ((*iter)->id_ != 0 && (*iter2)->id_ == 0)
            (*iter2)->id_ = (*iter)->id_;
        }
    }
  }
  std::vector<int> merge_flag;

  for (size_t i = 0; i < origin_list.size(); i++) {
    auto it =
        std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id_);
    if (it != merge_flag.end()) continue;
    if (origin_list[i]->id_ == 0) {
      merge_plane_list.push_back(origin_list[i]);
      continue;
    }
    std::shared_ptr<Plane> merge_plane(new Plane);
    (*merge_plane) = (*origin_list[i]);
    bool is_merge = false;
    for (size_t j = 0; j < origin_list.size(); j++) {
      if (i == j) continue;
      if (origin_list[j]->id_ == origin_list[i]->id_) {
        is_merge = true;
        Eigen::Matrix3d P_PT1 =
            (merge_plane->covariance_ +
             merge_plane->center_ * merge_plane->center_.transpose()) *
            merge_plane->points_size_;
        Eigen::Matrix3d P_PT2 =
            (origin_list[j]->covariance_ +
             origin_list[j]->center_ * origin_list[j]->center_.transpose()) *
            origin_list[j]->points_size_;
        Eigen::Vector3d merge_center =
            (merge_plane->center_ * merge_plane->points_size_ +
             origin_list[j]->center_ * origin_list[j]->points_size_) /
            (merge_plane->points_size_ + origin_list[j]->points_size_);
        Eigen::Matrix3d merge_covariance =
            (P_PT1 + P_PT2) /
                (merge_plane->points_size_ + origin_list[j]->points_size_) -
            merge_center * merge_center.transpose();
        merge_plane->covariance_ = merge_covariance;
        merge_plane->center_ = merge_center;
        merge_plane->points_size_ =
            merge_plane->points_size_ + origin_list[j]->points_size_;
        merge_plane->sub_plane_num_ += origin_list[j]->sub_plane_num_;
        // for (size_t k = 0; k < origin_list[j]->cloud.size(); k++) {
        //   merge_plane->cloud.points.push_back(origin_list[j]->cloud.points[k]);
        // }
        Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance_);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3f::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        merge_plane->normal_ << evecs.real()(0, evalsMin),
            evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
        merge_plane->radius_ = sqrt(evalsReal(evalsMax));
        merge_plane->d_ = -(merge_plane->normal_(0) * merge_plane->center_(0) +
                            merge_plane->normal_(1) * merge_plane->center_(1) +
                            merge_plane->normal_(2) * merge_plane->center_(2));
        merge_plane->p_center_.x = merge_plane->center_(0);
        merge_plane->p_center_.y = merge_plane->center_(1);
        merge_plane->p_center_.z = merge_plane->center_(2);
        merge_plane->p_center_.normal_x = merge_plane->normal_(0);
        merge_plane->p_center_.normal_y = merge_plane->normal_(1);
        merge_plane->p_center_.normal_z = merge_plane->normal_(2);
      }
    }
    if (is_merge) {
      merge_flag.push_back(merge_plane->id_);
      merge_plane_list.push_back(merge_plane);
    }
  }
}

void SemanticTriangularDescManager::binary_extractor_semantic(
    const std::vector<std::shared_ptr<Plane>> proj_plane_list,
    const pcl::PointCloud<pcl::PointXYZL>::Ptr &input_cloud,
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map,
    std::vector<BinaryDescriptor> &binary_descriptor_list) {
  binary_descriptor_list.clear();
  std::vector<BinaryDescriptor> temp_binary_list;
  Eigen::Vector3d last_normal(0, 0, 0);
  int useful_proj_num = 0;
  for (int i = 0; i < proj_plane_list.size(); i++) {
    std::vector<BinaryDescriptor> prepare_binary_list;
    Eigen::Vector3d proj_center = proj_plane_list[i]->center_;
    Eigen::Vector3d proj_normal = proj_plane_list[i]->normal_;
    if (proj_normal.z() < 0) {
      proj_normal = -proj_normal;
    }
    if ((proj_normal - last_normal).norm() < 0.3 ||
        (proj_normal + last_normal).norm() > 0.3) {
      last_normal = proj_normal;
      if (print_debug_info_) {
        std::cout << "[Description] reference plane normal:"
                  << proj_normal.transpose()
                  << ", center:" << proj_center.transpose() << std::endl;
      }
      useful_proj_num++;
      extract_binary_semantic(proj_center, proj_normal, input_cloud, voxel_map,
                     prepare_binary_list);
      for (auto bi : prepare_binary_list) {
        temp_binary_list.push_back(bi);
      }
      if (useful_proj_num == config_setting_.proj_plane_num_) {
        break;
      }
    }
  }
  non_maxi_suppression(temp_binary_list);
  if (config_setting_.useful_corner_num_ > temp_binary_list.size()) {
    binary_descriptor_list = temp_binary_list;
  } else {
    std::sort(temp_binary_list.begin(), temp_binary_list.end(),
              binary_greater_sort);
    for (size_t i = 0; i < config_setting_.useful_corner_num_; i++) {
      binary_descriptor_list.push_back(temp_binary_list[i]);
    }
  }
  return;
}

void BtcDescManager::binary_extractor(
    const std::vector<std::shared_ptr<Plane>> proj_plane_list,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<BinaryDescriptor> &binary_descriptor_list) {
  binary_descriptor_list.clear();
  std::vector<BinaryDescriptor> temp_binary_list;
  Eigen::Vector3d last_normal(0, 0, 0);
  int useful_proj_num = 0;
  for (int i = 0; i < proj_plane_list.size(); i++) {
    std::vector<BinaryDescriptor> prepare_binary_list;
    Eigen::Vector3d proj_center = proj_plane_list[i]->center_;
    Eigen::Vector3d proj_normal = proj_plane_list[i]->normal_;
    if (proj_normal.z() < 0) {
      proj_normal = -proj_normal;
    }
    if ((proj_normal - last_normal).norm() < 0.3 ||
        (proj_normal + last_normal).norm() > 0.3) {
      last_normal = proj_normal;
      std::cout << "[Description] reference plane normal:"
                << proj_normal.transpose()
                << ", center:" << proj_center.transpose() << std::endl;
      useful_proj_num++;
      extract_binary(proj_center, proj_normal, input_cloud,
                     prepare_binary_list);
      for (auto bi : prepare_binary_list) {
        temp_binary_list.push_back(bi);
      }
      if (useful_proj_num == config_setting_.proj_plane_num_) {
        break;
      }
    }
  }
  non_maxi_suppression(temp_binary_list);
  if (config_setting_.useful_corner_num_ > temp_binary_list.size()) {
    binary_descriptor_list = temp_binary_list;
  } else {
    std::sort(temp_binary_list.begin(), temp_binary_list.end(),
              binary_greater_sort);
    for (size_t i = 0; i < config_setting_.useful_corner_num_; i++) {
      binary_descriptor_list.push_back(temp_binary_list[i]);
    }
  }
  return;
}

void BtcDescManager::extract_binary(
    const Eigen::Vector3d &project_center,
    const Eigen::Vector3d &project_normal,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<BinaryDescriptor> &binary_list) {
  binary_list.clear();
  double binary_min_dis = config_setting_.summary_min_thre_;
  double resolution = config_setting_.proj_image_resolution_;
  double dis_threshold_min = config_setting_.proj_dis_min_;
  double dis_threshold_max = config_setting_.proj_dis_max_;
  double high_inc = config_setting_.proj_image_high_inc_;
  bool line_filter_enable = config_setting_.line_filter_enable_;
  double A = project_normal[0];
  double B = project_normal[1];
  double C = project_normal[2];
  double D =
      -(A * project_center[0] + B * project_center[1] + C * project_center[2]);
  std::vector<Eigen::Vector3d> projection_points;
  // Eigen::Vector3d x_axis(1, 1, 0);
  Eigen::Vector3d x_axis(1, 0, 0);
  if (C != 0) {
    x_axis[2] = -(A + B) / C;
  } else if (B != 0) {
    x_axis[1] = -A / B;
  } else {
    x_axis[0] = 0;
    x_axis[1] = 1;
  }
  x_axis.normalize();
  Eigen::Vector3d y_axis = project_normal.cross(x_axis);
  y_axis.normalize();
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx = -(ax * project_center[0] + bx * project_center[1] +
                cx * project_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy = -(ay * project_center[0] + by * project_center[1] +
                cy * project_center[2]);
  std::vector<Eigen::Vector2d> point_list_2d;
  pcl::PointCloud<pcl::PointXYZ> point_list_3d;
  std::vector<double> dis_list_2d;
  for (size_t i = 0; i < input_cloud->size(); i++) {
    double x = input_cloud->points[i].x;
    double y = input_cloud->points[i].y;
    double z = input_cloud->points[i].z;
    double dis = x * A + y * B + z * C + D;
    pcl::PointXYZ pi;
    if (dis < dis_threshold_min || dis > dis_threshold_max) {
      continue;
    } else {
      if (dis > dis_threshold_min && dis <= dis_threshold_max) {
        pi.x = x;
        pi.y = y;
        pi.z = z;
      }
    }
    Eigen::Vector3d cur_project;

    cur_project[0] = (-A * (B * y + C * z + D) + x * (B * B + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[1] = (-B * (A * x + C * z + D) + y * (A * A + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[2] = (-C * (A * x + B * y + D) + z * (A * A + B * B)) /
                     (A * A + B * B + C * C);
    pcl::PointXYZ p;
    p.x = cur_project[0];
    p.y = cur_project[1];
    p.z = cur_project[2];
    double project_x =
        cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
    double project_y =
        cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
    Eigen::Vector2d p_2d(project_x, project_y);
    point_list_2d.push_back(p_2d);
    dis_list_2d.push_back(dis);
    point_list_3d.points.push_back(pi);
  }
  double min_x = 10;
  double max_x = -10;
  double min_y = 10;
  double max_y = -10;
  if (point_list_2d.size() <= 5) {
    return;
  }
  for (auto pi : point_list_2d) {
    if (pi[0] < min_x) {
      min_x = pi[0];
    }
    if (pi[0] > max_x) {
      max_x = pi[0];
    }
    if (pi[1] < min_y) {
      min_y = pi[1];
    }
    if (pi[1] > max_y) {
      max_y = pi[1];
    }
  }
  // segment project cloud
  int segmen_base_num = 5;
  double segmen_len = segmen_base_num * resolution;
  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);

  std::vector<double> **dis_container = new std::vector<double> *[x_axis_len];
  BinaryDescriptor **binary_container = new BinaryDescriptor *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_container[i] = new std::vector<double>[y_axis_len];
    binary_container[i] = new BinaryDescriptor[y_axis_len];
  }
  double **img_count = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    img_count[i] = new double[y_axis_len];
  }
  double **dis_array = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_array[i] = new double[y_axis_len];
  }
  double **mean_x_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_x_list[i] = new double[y_axis_len];
  }
  double **mean_y_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_y_list[i] = new double[y_axis_len];
  }
  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      img_count[x][y] = 0;
      mean_x_list[x][y] = 0;
      mean_y_list[x][y] = 0;
      dis_array[x][y] = 0;
      std::vector<double> single_dis_container;
      dis_container[x][y] = single_dis_container;
    }
  }

  for (size_t i = 0; i < point_list_2d.size(); i++) {
    int x_index = (int)((point_list_2d[i][0] - min_x) / resolution);
    int y_index = (int)((point_list_2d[i][1] - min_y) / resolution);
    mean_x_list[x_index][y_index] += point_list_2d[i][0];
    mean_y_list[x_index][y_index] += point_list_2d[i][1];
    img_count[x_index][y_index]++;
    dis_container[x_index][y_index].push_back(dis_list_2d[i]);
  }

  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      // calc segment dis array
      if (img_count[x][y] > 0) {
        int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
        std::vector<bool> occup_list;
        std::vector<double> cnt_list;
        BinaryDescriptor single_binary;
        for (size_t i = 0; i < cut_num; i++) {
          cnt_list.push_back(0);
          occup_list.push_back(false);
        }
        for (size_t j = 0; j < dis_container[x][y].size(); j++) {
          int cnt_index =
              (dis_container[x][y][j] - dis_threshold_min) / high_inc;
          cnt_list[cnt_index]++;
        }
        double segmnt_dis = 0;
        for (size_t i = 0; i < cut_num; i++) {
          if (cnt_list[i] >= 1) {
            segmnt_dis++;
            occup_list[i] = true;
          }
        }
        dis_array[x][y] = segmnt_dis;
        single_binary.occupy_array_ = occup_list;
        single_binary.summary_ = segmnt_dis;
        binary_container[x][y] = single_binary;
      }
    }
  }

  // filter by distance
  std::vector<double> max_dis_list;
  std::vector<int> max_dis_x_index_list;
  std::vector<int> max_dis_y_index_list;

  for (int x_segment_index = 0; x_segment_index < x_segment_num;
       x_segment_index++) {
    for (int y_segment_index = 0; y_segment_index < y_segment_num;
         y_segment_index++) {
      double max_dis = 0;
      int max_dis_x_index = -10;
      int max_dis_y_index = -10;
      for (int x_index = x_segment_index * segmen_base_num;
           x_index < (x_segment_index + 1) * segmen_base_num; x_index++) {
        for (int y_index = y_segment_index * segmen_base_num;
             y_index < (y_segment_index + 1) * segmen_base_num; y_index++) {
          if (dis_array[x_index][y_index] > max_dis) {
            max_dis = dis_array[x_index][y_index];
            max_dis_x_index = x_index;
            max_dis_y_index = y_index;
          }
        }
      }
      if (max_dis >= binary_min_dis) {
        max_dis_list.push_back(max_dis);
        max_dis_x_index_list.push_back(max_dis_x_index);
        max_dis_y_index_list.push_back(max_dis_y_index);
      }
    }
  }
  // calc line or not
  std::vector<Eigen::Vector2i> direction_list;
  Eigen::Vector2i d(0, 1);
  direction_list.push_back(d);
  d << 1, 0;
  direction_list.push_back(d);
  d << 1, 1;
  direction_list.push_back(d);
  d << 1, -1;
  direction_list.push_back(d);
  for (size_t i = 0; i < max_dis_list.size(); i++) {
    Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
    if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
        p[1] >= y_axis_len - 1) {
      continue;
    }
    bool is_add = true;

    if (line_filter_enable) {
      for (int j = 0; j < 4; j++) {
        Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
        if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
            p[1] >= y_axis_len - 1) {
          continue;
        }
        Eigen::Vector2i p1 = p + direction_list[j];
        Eigen::Vector2i p2 = p - direction_list[j];
        double threshold = dis_array[p[0]][p[1]] - 3;
        if (dis_array[p1[0]][p1[1]] >= threshold) {
          if (dis_array[p2[0]][p2[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold) {
          if (dis_array[p1[0]][p1[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
            is_add = false;
          }
        }
        if (dis_array[p1[0]][p1[1]] >= threshold) {
          if (dis_array[p2[0]][p2[1]] >= threshold) {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold) {
          if (dis_array[p1[0]][p1[1]] >= threshold) {
            is_add = false;
          }
        }
      }
    }
    if (is_add) {
      double px =
          mean_x_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      double py =
          mean_y_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      Eigen::Vector3d coord = py * x_axis + px * y_axis + project_center;
      pcl::PointXYZ pi;
      pi.x = coord[0];
      pi.y = coord[1];
      pi.z = coord[2];
      BinaryDescriptor single_binary =
          binary_container[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      single_binary.location_ = coord;
      binary_list.push_back(single_binary);
    }
  }
  for (int i = 0; i < x_axis_len; i++) {
    delete[] binary_container[i];
    delete[] dis_container[i];
    delete[] img_count[i];
    delete[] dis_array[i];
    delete[] mean_x_list[i];
    delete[] mean_y_list[i];
  }
  delete[] binary_container;
  delete[] dis_container;
  delete[] img_count;
  delete[] dis_array;
  delete[] mean_x_list;
  delete[] mean_y_list;
}

// 从位置获取语义标签（获取该体素中最频繁的标签）
uint32_t SemanticTriangularDescManager::get_semantic_label_at_location(
    const Eigen::Vector3d &location,
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map) {
  // 计算位置所在的体素位置
  double loc_xyz[3];
  loc_xyz[0] = location[0] / config_setting_.voxel_size_;
  loc_xyz[1] = location[1] / config_setting_.voxel_size_;
  loc_xyz[2] = location[2] / config_setting_.voxel_size_;
  for (int j = 0; j < 3; j++) {
    if (loc_xyz[j] < 0) {
      loc_xyz[j] -= 1.0;
    }
  }
  VOXEL_LOC voxel_pos((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  
  // 查找对应的体素
  auto iter = voxel_map.find(voxel_pos);
  if (iter != voxel_map.end() && !iter->second.empty()) {
    // 统计每个标签的数量
    std::map<uint32_t, int> label_count;
    for (const auto& point_label : iter->second) {
      uint32_t label = point_label.second & 0xFFFF;
      label_count[label]++;
    }
    
    // 返回最频繁的标签
    uint32_t most_frequent_label = 0;
    int max_count = 0;
    for (const auto& pair : label_count) {
      if (pair.second > max_count) {
        max_count = pair.second;
        most_frequent_label = pair.first;
      }
    }
    return most_frequent_label;
  }
  
  return 0;  // 默认返回0（无标签）
}

double SemanticTriangularDescManager::get_semantic_ratio(const Eigen::Vector3d &location, uint32_t semantic_label,
                            const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map) {
  // 计算位置所在的体素位置
  double loc_xyz[3];
  loc_xyz[0] = location[0] / config_setting_.voxel_size_;
  loc_xyz[1] = location[1] / config_setting_.voxel_size_;
  loc_xyz[2] = location[2] / config_setting_.voxel_size_;
  for (int j = 0; j < 3; j++) {
    if (loc_xyz[j] < 0) {
      loc_xyz[j] -= 1.0;
    }
  }
  VOXEL_LOC voxel_pos((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  
  // 查找对应的体素
  auto iter = voxel_map.find(voxel_pos);
  if (iter != voxel_map.end()) {
    // 统计该语义标签在体素中的数量
    int label_count = 0;
    int total_count = iter->second.size();
    for (const auto& point_label : iter->second) {
      uint32_t label = point_label.second & 0xFFFF;
      if (label == semantic_label) {
        label_count++;
      }
    }
    // 返回比例
    return (total_count > 0) ? static_cast<double>(label_count) / total_count : 0.0;
  }
  
  // 如果找不到对应的体素，返回0（表示比例未知）
  return 0.0;
}

// 获取位置处的第一和第二语义标签及其比例
void SemanticTriangularDescManager::get_semantic_labels_at_location(
    const Eigen::Vector3d &location,
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map,
    uint32_t &label_1, double &ratio_1, uint32_t &label_2, double &ratio_2) {
  // 初始化返回值
  label_1 = 0;
  ratio_1 = 0.0;
  label_2 = 0;
  ratio_2 = 0.0;
  
  // 计算位置所在的体素位置
  double loc_xyz[3];
  loc_xyz[0] = location[0] / config_setting_.voxel_size_;
  loc_xyz[1] = location[1] / config_setting_.voxel_size_;
  loc_xyz[2] = location[2] / config_setting_.voxel_size_;
  for (int j = 0; j < 3; j++) {
    if (loc_xyz[j] < 0) {
      loc_xyz[j] -= 1.0;
    }
  }
  VOXEL_LOC voxel_pos((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  
  // 查找对应的体素
  auto iter = voxel_map.find(voxel_pos);
  if (iter != voxel_map.end() && !iter->second.empty()) {
    // 统计每个标签的数量
    std::map<uint32_t, int> label_count;
    int total_count = iter->second.size();
    for (const auto& point_label : iter->second) {
      uint32_t label = point_label.second & 0xFFFF;
      label_count[label]++;
    }
    
    // 找到频率最高和第二高的标签
    uint32_t most_frequent_label = 0;
    uint32_t second_frequent_label = 0;
    int max_count = 0;
    int second_max_count = 0;
    
    for (const auto& pair : label_count) {
      if (pair.second > max_count) {
        second_max_count = max_count;
        second_frequent_label = most_frequent_label;
        max_count = pair.second;
        most_frequent_label = pair.first;
      } else if (pair.second > second_max_count) {
        second_max_count = pair.second;
        second_frequent_label = pair.first;
      }
    }
    
    // 返回第一和第二标签及其比例
    label_1 = most_frequent_label;
    ratio_1 = (total_count > 0) ? static_cast<double>(max_count) / total_count : 0.0;
    label_2 = second_frequent_label;
    ratio_2 = (total_count > 0) ? static_cast<double>(second_max_count) / total_count : 0.0;
  }
}

void SemanticTriangularDescManager::extract_binary_semantic(
    const Eigen::Vector3d &project_center,
    const Eigen::Vector3d &project_normal,
    const pcl::PointCloud<pcl::PointXYZL>::Ptr &input_cloud,
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map,
    std::vector<BinaryDescriptor> &binary_list) {
  // 基本实现与extract_binary相同，但需要处理语义标签
  // 这里先复制extract_binary的逻辑，然后添加语义标签处理
  binary_list.clear();
  double binary_min_dis = config_setting_.summary_min_thre_;
  double resolution = config_setting_.proj_image_resolution_;
  double dis_threshold_min = config_setting_.proj_dis_min_;
  double dis_threshold_max = config_setting_.proj_dis_max_;
  double high_inc = config_setting_.proj_image_high_inc_;
  bool line_filter_enable = config_setting_.line_filter_enable_;
  double A = project_normal[0];
  double B = project_normal[1];
  double C = project_normal[2];
  double D =
      -(A * project_center[0] + B * project_center[1] + C * project_center[2]);
  std::vector<Eigen::Vector3d> projection_points;
  Eigen::Vector3d x_axis(1, 0, 0);
  if (C != 0) {
    x_axis[2] = -(A + B) / C;
  } else if (B != 0) {
    x_axis[1] = -A / B;
  } else {
    x_axis[0] = 0;
    x_axis[1] = 1;
  }
  x_axis.normalize();
  Eigen::Vector3d y_axis = project_normal.cross(x_axis);
  y_axis.normalize();
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx = -(ax * project_center[0] + bx * project_center[1] +
                cx * project_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy = -(ay * project_center[0] + by * project_center[1] +
                cy * project_center[2]);
  std::vector<Eigen::Vector2d> point_list_2d;
  pcl::PointCloud<pcl::PointXYZ> point_list_3d;
  std::vector<double> dis_list_2d;
  std::vector<uint32_t> label_list;  // 添加标签列表
  for (size_t i = 0; i < input_cloud->size(); i++) {
    double x = input_cloud->points[i].x;
    double y = input_cloud->points[i].y;
    double z = input_cloud->points[i].z;
    double dis = x * A + y * B + z * C + D;
    pcl::PointXYZ pi;
    if (dis < dis_threshold_min || dis > dis_threshold_max) {
      continue;
    } else {
      if (dis > dis_threshold_min && dis <= dis_threshold_max) {
        pi.x = x;
        pi.y = y;
        pi.z = z;
      }
    }
    Eigen::Vector3d cur_project;

    cur_project[0] = (-A * (B * y + C * z + D) + x * (B * B + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[1] = (-B * (A * x + C * z + D) + y * (A * A + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[2] = (-C * (A * x + B * y + D) + z * (A * A + B * B)) /
                     (A * A + B * B + C * C);
    pcl::PointXYZ p;
    p.x = cur_project[0];
    p.y = cur_project[1];
    p.z = cur_project[2];
    double project_x =
        cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
    double project_y =
        cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
    Eigen::Vector2d p_2d(project_x, project_y);
    point_list_2d.push_back(p_2d);
    dis_list_2d.push_back(dis);
    point_list_3d.points.push_back(pi);
    label_list.push_back(input_cloud->points[i].label & 0xFFFF);  // 保存标签
  }
  double min_x = 10;
  double max_x = -10;
  double min_y = 10;
  double max_y = -10;
  if (point_list_2d.size() <= 5) {
    if (print_debug_info_) {
      std::cerr << "[extract_binary_semantic] Warning: Too few points in projection ("
                << point_list_2d.size() << " <= 5), skipping!" << std::endl;
      std::cerr << "[extract_binary_semantic] Debug: dis_threshold_min=" << dis_threshold_min 
                << ", dis_threshold_max=" << dis_threshold_max << std::endl;
    }
    return;
  }
  
  if (print_debug_info_) {
    std::cerr << "[extract_binary_semantic] Debug: point_list_2d size: " << point_list_2d.size() << std::endl;
  }
  
  for (auto pi : point_list_2d) {
    if (pi[0] < min_x) {
      min_x = pi[0];
    }
    if (pi[0] > max_x) {
      max_x = pi[0];
    }
    if (pi[1] < min_y) {
      min_y = pi[1];
    }
    if (pi[1] > max_y) {
      max_y = pi[1];
    }
  }
  // segment project cloud
  int segmen_base_num = 5;
  double segmen_len = segmen_base_num * resolution;
  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);

  std::vector<double> **dis_container = new std::vector<double> *[x_axis_len];
  BinaryDescriptor **binary_container = new BinaryDescriptor *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_container[i] = new std::vector<double>[y_axis_len];
    binary_container[i] = new BinaryDescriptor[y_axis_len];
  }
  double **img_count = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    img_count[i] = new double[y_axis_len];
  }
  double **dis_array = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_array[i] = new double[y_axis_len];
  }
  double **mean_x_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_x_list[i] = new double[y_axis_len];
  }
  double **mean_y_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_y_list[i] = new double[y_axis_len];
  }
  // 语义标签统计（整格）及每格点的标签序列（与 dis_container 同序，用于按 bin 建语义描述子）
  std::unordered_map<uint32_t, int> **label_count_map = new std::unordered_map<uint32_t, int> *[x_axis_len];
  std::vector<uint32_t> **label_container = new std::vector<uint32_t> *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    label_count_map[i] = new std::unordered_map<uint32_t, int>[y_axis_len];
    label_container[i] = new std::vector<uint32_t>[y_axis_len];
  }
  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      img_count[x][y] = 0;
      mean_x_list[x][y] = 0;
      mean_y_list[x][y] = 0;
      dis_array[x][y] = 0;
      std::vector<double> single_dis_container;
      dis_container[x][y] = single_dis_container;
    }
  }

  for (size_t i = 0; i < point_list_2d.size(); i++) {
    int x_index = (int)((point_list_2d[i][0] - min_x) / resolution);
    int y_index = (int)((point_list_2d[i][1] - min_y) / resolution);
    mean_x_list[x_index][y_index] += point_list_2d[i][0];
    mean_y_list[x_index][y_index] += point_list_2d[i][1];
    img_count[x_index][y_index]++;
    dis_container[x_index][y_index].push_back(dis_list_2d[i]);
    label_container[x_index][y_index].push_back(label_list[i]);
    label_count_map[x_index][y_index][label_list[i]]++;
  }

  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      // calc segment dis array
      if (img_count[x][y] > 0) {
        int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
        std::vector<bool> occup_list;
        std::vector<double> cnt_list;
        BinaryDescriptor single_binary;
        for (size_t i = 0; i < cut_num; i++) {
          cnt_list.push_back(0);
          occup_list.push_back(false);
        }
        for (size_t j = 0; j < dis_container[x][y].size(); j++) {
          int cnt_index =
              (dis_container[x][y][j] - dis_threshold_min) / high_inc;
          cnt_list[cnt_index]++;
        }
        double segmnt_dis = 0;
        for (size_t i = 0; i < cut_num; i++) {
          if (cnt_list[i] >= 1) {
            segmnt_dis++;
            occup_list[i] = true;
          }
        }
        dis_array[x][y] = segmnt_dis;
        single_binary.occupy_array_ = occup_list;
        single_binary.summary_ = segmnt_dis;
        // 语义描述子：按与 occupy_array_ 同序，每格（距离 bin）取该格内最多语义标签及比例
        single_binary.semantic_label_array_.clear();
        single_binary.semantic_ratio_array_.clear();
        single_binary.semantic_label_array_.resize(cut_num, 0);
        single_binary.semantic_ratio_array_.resize(cut_num, 0.0);
        for (int bin_k = 0; bin_k < cut_num; bin_k++) {
          std::unordered_map<uint32_t, int> bin_label_count;
          for (size_t j = 0; j < dis_container[x][y].size(); j++) {
            int cnt_index = (dis_container[x][y][j] - dis_threshold_min) / high_inc;
            if (cnt_index == bin_k) {
              uint32_t lab = label_container[x][y][j];
              bin_label_count[lab]++;
            }
          }
          int bin_total = 0;
          int bin_max_count = 0;
          uint32_t bin_majority = 0;
          for (const auto& p : bin_label_count) {
            bin_total += p.second;
            if (p.second > bin_max_count) {
              bin_max_count = p.second;
              bin_majority = p.first;
            }
          }
          single_binary.semantic_label_array_[bin_k] = bin_majority;
          single_binary.semantic_ratio_array_[bin_k] = (bin_total > 0) ? static_cast<double>(bin_max_count) / bin_total : 0.0;
        }
        // 保留整格最多/次多标签用于兼容或可视化
        uint32_t most_frequent_label = 0;
        uint32_t second_frequent_label = 0;
        int max_count = 0;
        int second_max_count = 0;
        for (const auto& label_pair : label_count_map[x][y]) {
          if (label_pair.second > max_count) {
            second_max_count = max_count;
            second_frequent_label = most_frequent_label;
            max_count = label_pair.second;
            most_frequent_label = label_pair.first;
          } else if (label_pair.second > second_max_count) {
            second_max_count = label_pair.second;
            second_frequent_label = label_pair.first;
          }
        }
        single_binary.semantic_label_ = most_frequent_label;
        single_binary.semantic_ratio_ = (img_count[x][y] > 0) ? static_cast<double>(max_count) / img_count[x][y] : 0.0;
        single_binary.semantic_label_2_ = second_frequent_label;
        single_binary.semantic_ratio_2_ = (img_count[x][y] > 0) ? static_cast<double>(second_max_count) / img_count[x][y] : 0.0;
        binary_container[x][y] = single_binary;
      }
    }
  }

  // filter by distance
  std::vector<double> max_dis_list;
  std::vector<int> max_dis_x_index_list;
  std::vector<int> max_dis_y_index_list;
  
  int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
  double max_segmt_dis_found = 0;
  int candidates_before_filter = 0;

  for (int x_segment_index = 0; x_segment_index < x_segment_num;
       x_segment_index++) {
    for (int y_segment_index = 0; y_segment_index < y_segment_num;
         y_segment_index++) {
      double max_dis = 0;
      int max_dis_x_index = -10;
      int max_dis_y_index = -10;
      for (int x_index = x_segment_index * segmen_base_num;
           x_index < (x_segment_index + 1) * segmen_base_num; x_index++) {
        for (int y_index = y_segment_index * segmen_base_num;
             y_index < (y_segment_index + 1) * segmen_base_num; y_index++) {
          if (dis_array[x_index][y_index] > max_dis) {
            max_dis = dis_array[x_index][y_index];
            max_dis_x_index = x_index;
            max_dis_y_index = y_index;
          }
        }
      }
      if (max_dis > 0) {
        candidates_before_filter++;
        max_segmt_dis_found = std::max(max_segmt_dis_found, max_dis);
      }
      if (max_dis >= binary_min_dis) {
        max_dis_list.push_back(max_dis);
        max_dis_x_index_list.push_back(max_dis_x_index);
        max_dis_y_index_list.push_back(max_dis_y_index);
      }
    }
  }
  
  if (print_debug_info_) {
    std::cerr << "[extract_binary_semantic] Debug: cut_num=" << cut_num 
              << ", binary_min_dis=" << binary_min_dis 
              << ", max_segmt_dis_found=" << max_segmt_dis_found
              << ", candidates_before_filter=" << candidates_before_filter
              << ", candidates_after_filter=" << max_dis_list.size() << std::endl;
  }
  
  // calc line or not
  std::vector<Eigen::Vector2i> direction_list;
  Eigen::Vector2i d(0, 1);
  direction_list.push_back(d);
  d << 1, 0;
  direction_list.push_back(d);
  d << 1, 1;
  direction_list.push_back(d);
  d << 1, -1;
  direction_list.push_back(d);
  
  int before_line_filter = max_dis_list.size();
  int edge_filtered = 0;
  
  for (size_t i = 0; i < max_dis_list.size(); i++) {
    Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
    if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
        p[1] >= y_axis_len - 1) {
      edge_filtered++;
      continue;
    }
    bool is_add = true;

    if (line_filter_enable) {
      for (int j = 0; j < 4; j++) {
        Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
        if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
            p[1] >= y_axis_len - 1) {
          continue;
        }
        Eigen::Vector2i p1 = p + direction_list[j];
        Eigen::Vector2i p2 = p - direction_list[j];
        double threshold = dis_array[p[0]][p[1]] - 3;
        if (dis_array[p1[0]][p1[1]] >= threshold) {
          if (dis_array[p2[0]][p2[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold) {
          if (dis_array[p1[0]][p1[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
            is_add = false;
          }
        }
        if (dis_array[p1[0]][p1[1]] >= threshold) {
          if (dis_array[p2[0]][p2[1]] >= threshold) {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold) {
          if (dis_array[p1[0]][p1[1]] >= threshold) {
            is_add = false;
          }
        }
      }
    }
    if (is_add) {
      double px =
          mean_x_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      double py =
          mean_y_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      Eigen::Vector3d coord = py * x_axis + px * y_axis + project_center;
      pcl::PointXYZ pi;
      pi.x = coord[0];
      pi.y = coord[1];
      pi.z = coord[2];
      BinaryDescriptor single_binary =
          binary_container[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      single_binary.location_ = coord;
      // 更新语义标签比例（从体素图中获取更准确的值）
      single_binary.semantic_ratio_ = get_semantic_ratio(coord, single_binary.semantic_label_, voxel_map);
      // 同时更新第二标签的比例
      if (single_binary.semantic_label_2_ != 0) {
        single_binary.semantic_ratio_2_ = get_semantic_ratio(coord, single_binary.semantic_label_2_, voxel_map);
      }
      binary_list.push_back(single_binary);
    }
  }
  
  if (print_debug_info_) {
    std::cerr << "[extract_binary_semantic] Debug: After line filter, binary_list size: " 
              << binary_list.size() 
              << ", edge_filtered=" << edge_filtered
              << ", before_line_filter=" << before_line_filter << std::endl;
  }
  
  for (int i = 0; i < x_axis_len; i++) {
    delete[] binary_container[i];
    delete[] dis_container[i];
    delete[] img_count[i];
    delete[] dis_array[i];
    delete[] mean_x_list[i];
    delete[] mean_y_list[i];
    delete[] label_count_map[i];
    delete[] label_container[i];
  }
  delete[] binary_container;
  delete[] dis_container;
  delete[] img_count;
  delete[] dis_array;
  delete[] mean_x_list;
  delete[] mean_y_list;
  delete[] label_count_map;
  delete[] label_container;
}

void BtcDescManager::non_maxi_suppression(
    std::vector<BinaryDescriptor> &binary_list) {
  // 检查二进制描述子列表是否为空
  if (binary_list.empty()) {
    return;
  }
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr prepare_key_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  std::vector<int> pre_count_list;
  std::vector<bool> is_add_list;
  for (auto var : binary_list) {
    pcl::PointXYZ pi;
    pi.x = var.location_[0];
    pi.y = var.location_[1];
    pi.z = var.location_[2];
    prepare_key_cloud->push_back(pi);
    pre_count_list.push_back(var.summary_);
    is_add_list.push_back(true);
  }
  
  // 检查prepare_key_cloud是否为空（虽然理论上不应该，但为了安全）
  if (prepare_key_cloud->empty()) {
    if (print_debug_info_) {
      std::cerr << "[non_maxi_suppression] Warning: prepare_key_cloud is empty!" << std::endl;
    }
    return;
  }
  
  kd_tree.setInputCloud(prepare_key_cloud);
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  double radius = config_setting_.non_max_suppression_radius_;
  
  if (print_debug_info_) {
    std::cerr << "[non_maxi_suppression] Debug: Using radius=" << radius 
              << " for NMS, input binary_list size=" << binary_list.size() << std::endl;
  }
  
  for (size_t i = 0; i < prepare_key_cloud->size(); i++) {
    pcl::PointXYZ searchPoint = prepare_key_cloud->points[i];
    if (kd_tree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch,
                             pointRadiusSquaredDistance) > 0) {
      Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
      for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
        Eigen::Vector3d pj(
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].x,
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].y,
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].z);
        if (pointIdxRadiusSearch[j] == i) {
          continue;
        }
        // 非极大值抑制：如果当前点的summary小于邻居，则抑制当前点
        // 如果summary相等，保留索引较小的点（避免互相抑制导致都删除）
        if (pre_count_list[i] < pre_count_list[pointIdxRadiusSearch[j]] ||
            (pre_count_list[i] == pre_count_list[pointIdxRadiusSearch[j]] && 
             i > pointIdxRadiusSearch[j])) {
          is_add_list[i] = false;
          // 注意：不能break，因为需要检查所有邻居来确定是否应该抑制
          // 但一旦确定需要抑制，可以继续检查其他邻居（不影响结果）
        }
      }
    }
  }
  std::vector<BinaryDescriptor> pass_binary_list;
  int suppressed_count = 0;
  for (size_t i = 0; i < is_add_list.size(); i++) {
    if (is_add_list[i]) {
      pass_binary_list.push_back(binary_list[i]);
    } else {
      suppressed_count++;
    }
  }
  
  if (print_debug_info_) {
    std::cerr << "[non_maxi_suppression] Debug: Suppressed " << suppressed_count 
              << " descriptors, remaining " << pass_binary_list.size() 
              << " descriptors (radius=" << radius << ")" << std::endl;
  }
  
  binary_list.clear();
  for (auto var : pass_binary_list) {
    binary_list.push_back(var);
  }
  return;
}

void BtcDescManager::generate_btc(
    const std::vector<BinaryDescriptor> &binary_list, const int &frame_id,
    std::vector<BTC> &btc_list) {
  btc_list.clear();
  
  // 检查二进制描述子列表是否为空
  if (binary_list.empty()) {
    if (print_debug_info_) {
      std::cerr << "[generate_btc] Warning: binary_list is empty!" << std::endl;
    }
    return;
  }
  
  double scale = 1.0 / config_setting_.std_side_resolution_;
  std::unordered_map<VOXEL_LOC, bool> feat_map;
  pcl::PointCloud<pcl::PointXYZ> key_cloud;
  for (auto var : binary_list) {
    pcl::PointXYZ pi;
    pi.x = var.location_[0];
    pi.y = var.location_[1];
    pi.z = var.location_[2];
    key_cloud.push_back(pi);
  }
  
  // 检查key_cloud是否为空（虽然理论上不应该，但为了安全）
  if (key_cloud.empty()) {
    if (print_debug_info_) {
      std::cerr << "[generate_btc] Warning: key_cloud is empty!" << std::endl;
    }
    return;
  }
  
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  kd_tree->setInputCloud(key_cloud.makeShared());
  int K = config_setting_.descriptor_near_num_;
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  for (size_t i = 0; i < key_cloud.size(); i++) {
    pcl::PointXYZ searchPoint = key_cloud.points[i];
    if (kd_tree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      for (int m = 1; m < K - 1; m++) {
        for (int n = m + 1; n < K; n++) {
          pcl::PointXYZ p1 = searchPoint;
          pcl::PointXYZ p2 = key_cloud.points[pointIdxNKNSearch[m]];
          pcl::PointXYZ p3 = key_cloud.points[pointIdxNKNSearch[n]];
          double a = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                          pow(p1.z - p2.z, 2));
          double b = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) +
                          pow(p1.z - p3.z, 2));
          double c = sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) +
                          pow(p3.z - p2.z, 2));
          if (a > config_setting_.descriptor_max_len_ ||
              b > config_setting_.descriptor_max_len_ ||
              c > config_setting_.descriptor_max_len_ ||
              a < config_setting_.descriptor_min_len_ ||
              b < config_setting_.descriptor_min_len_ ||
              c < config_setting_.descriptor_min_len_) {
            continue;
          }
          double temp;
          Eigen::Vector3d A, B, C;
          Eigen::Vector3i l1, l2, l3;
          Eigen::Vector3i l_temp;
          l1 << 1, 2, 0;
          l2 << 1, 0, 3;
          l3 << 0, 2, 3;
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (b > c) {
            temp = b;
            b = c;
            c = temp;
            l_temp = l2;
            l2 = l3;
            l3 = l_temp;
          }
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (fabs(c - (a + b)) < 0.2) {
            continue;
          }

          pcl::PointXYZ d_p;
          d_p.x = a * 1000;
          d_p.y = b * 1000;
          d_p.z = c * 1000;
          VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
          auto iter = feat_map.find(position);
          Eigen::Vector3d normal_1, normal_2, normal_3;
          BinaryDescriptor binary_A;
          BinaryDescriptor binary_B;
          BinaryDescriptor binary_C;
          if (iter == feat_map.end()) {
            if (l1[0] == l2[0]) {
              A << p1.x, p1.y, p1.z;
              binary_A = binary_list[i];
            } else if (l1[1] == l2[1]) {
              A << p2.x, p2.y, p2.z;
              binary_A = binary_list[pointIdxNKNSearch[m]];
            } else {
              A << p3.x, p3.y, p3.z;
              binary_A = binary_list[pointIdxNKNSearch[n]];
            }
            if (l1[0] == l3[0]) {
              B << p1.x, p1.y, p1.z;
              binary_B = binary_list[i];
            } else if (l1[1] == l3[1]) {
              B << p2.x, p2.y, p2.z;
              binary_B = binary_list[pointIdxNKNSearch[m]];
            } else {
              B << p3.x, p3.y, p3.z;
              binary_B = binary_list[pointIdxNKNSearch[n]];
            }
            if (l2[0] == l3[0]) {
              C << p1.x, p1.y, p1.z;
              binary_C = binary_list[i];
            } else if (l2[1] == l3[1]) {
              C << p2.x, p2.y, p2.z;
              binary_C = binary_list[pointIdxNKNSearch[m]];
            } else {
              C << p3.x, p3.y, p3.z;
              binary_C = binary_list[pointIdxNKNSearch[n]];
            }
            BTC single_descriptor;
            single_descriptor.binary_A_ = binary_A;
            single_descriptor.binary_B_ = binary_B;
            single_descriptor.binary_C_ = binary_C;
            single_descriptor.center_ = (A + B + C) / 3;
            single_descriptor.triangle_ << scale * a, scale * b, scale * c;
            single_descriptor.angle_[0] = fabs(5 * normal_1.dot(normal_2));
            single_descriptor.angle_[1] = fabs(5 * normal_1.dot(normal_3));
            single_descriptor.angle_[2] = fabs(5 * normal_3.dot(normal_2));
            // single_descriptor.angle << 0, 0, 0;
            single_descriptor.frame_number_ = frame_id;
            // 向后兼容：如果没有语义标签，设置为0
            single_descriptor.vertex_semantic_ << 0.0, 0.0, 0.0;
            single_descriptor.vertex_semantic_ratio_ << 0.0, 0.0, 0.0;
            // single_descriptor.score_frame_.push_back(frame_number);
            Eigen::Matrix3d triangle_positon;
            triangle_positon.block<3, 1>(0, 0) = A;
            triangle_positon.block<3, 1>(0, 1) = B;
            triangle_positon.block<3, 1>(0, 2) = C;
            // single_descriptor.position_list_.push_back(triangle_positon);
            // single_descriptor.triangle_scale_ = scale;
            feat_map[position] = true;
            btc_list.push_back(single_descriptor);
          }
        }
      }
    }
  }
}

void SemanticTriangularDescManager::generate_semantic_triangular_desc(
    const std::vector<BinaryDescriptor> &binary_list, const int &frame_id,
    std::vector<SemanticTriangularDescriptor> &std_list) {
  std_list.clear();
  
  // 检查二进制描述子列表是否为空
  if (binary_list.empty()) {
    if (print_debug_info_) {
      std::cerr << "[generate_semantic_triangular_desc] Warning: binary_list is empty!" << std::endl;
    }
    return;
  }
  
  double scale = 1.0 / config_setting_.std_side_resolution_;
  std::unordered_map<VOXEL_LOC, bool> feat_map;
  pcl::PointCloud<pcl::PointXYZ> key_cloud;
  for (auto var : binary_list) {
    pcl::PointXYZ pi;
    pi.x = var.location_[0];
    pi.y = var.location_[1];
    pi.z = var.location_[2];
    key_cloud.push_back(pi);
  }
  
  // 检查key_cloud是否为空（虽然理论上不应该，但为了安全）
  if (key_cloud.empty()) {
    if (print_debug_info_) {
      std::cerr << "[generate_semantic_triangular_desc] Warning: key_cloud is empty!" << std::endl;
    }
    return;
  }
  
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  kd_tree->setInputCloud(key_cloud.makeShared());
  int K = config_setting_.descriptor_near_num_;
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  for (size_t i = 0; i < key_cloud.size(); i++) {
    pcl::PointXYZ searchPoint = key_cloud.points[i];
    if (kd_tree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      for (int m = 1; m < K - 1; m++) {
        for (int n = m + 1; n < K; n++) {
          pcl::PointXYZ p1 = searchPoint;
          pcl::PointXYZ p2 = key_cloud.points[pointIdxNKNSearch[m]];
          pcl::PointXYZ p3 = key_cloud.points[pointIdxNKNSearch[n]];
          double a = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                          pow(p1.z - p2.z, 2));
          double b = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) +
                          pow(p1.z - p3.z, 2));
          double c = sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) +
                          pow(p3.z - p2.z, 2));
          if (a > config_setting_.descriptor_max_len_ ||
              b > config_setting_.descriptor_max_len_ ||
              c > config_setting_.descriptor_max_len_ ||
              a < config_setting_.descriptor_min_len_ ||
              b < config_setting_.descriptor_min_len_ ||
              c < config_setting_.descriptor_min_len_) {
            continue;
          }
          double temp;
          Eigen::Vector3d A, B, C;
          Eigen::Vector3i l1, l2, l3;
          Eigen::Vector3i l_temp;
          l1 << 1, 2, 0;
          l2 << 1, 0, 3;
          l3 << 0, 2, 3;
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (b > c) {
            temp = b;
            b = c;
            c = temp;
            l_temp = l2;
            l2 = l3;
            l3 = l_temp;
          }
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (fabs(c - (a + b)) < 0.2) {
            continue;
          }

          pcl::PointXYZ d_p;
          d_p.x = a * 1000;
          d_p.y = b * 1000;
          d_p.z = c * 1000;
          VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
          auto iter = feat_map.find(position);
          Eigen::Vector3d normal_1, normal_2, normal_3;
          BinaryDescriptor binary_A;
          BinaryDescriptor binary_B;
          BinaryDescriptor binary_C;
          if (iter == feat_map.end()) {
            if (l1[0] == l2[0]) {
              A << p1.x, p1.y, p1.z;
              binary_A = binary_list[i];
            } else if (l1[1] == l2[1]) {
              A << p2.x, p2.y, p2.z;
              binary_A = binary_list[pointIdxNKNSearch[m]];
            } else {
              A << p3.x, p3.y, p3.z;
              binary_A = binary_list[pointIdxNKNSearch[n]];
            }
            if (l1[0] == l3[0]) {
              B << p1.x, p1.y, p1.z;
              binary_B = binary_list[i];
            } else if (l1[1] == l3[1]) {
              B << p2.x, p2.y, p2.z;
              binary_B = binary_list[pointIdxNKNSearch[m]];
            } else {
              B << p3.x, p3.y, p3.z;
              binary_B = binary_list[pointIdxNKNSearch[n]];
            }
            if (l2[0] == l3[0]) {
              C << p1.x, p1.y, p1.z;
              binary_C = binary_list[i];
            } else if (l2[1] == l3[1]) {
              C << p2.x, p2.y, p2.z;
              binary_C = binary_list[pointIdxNKNSearch[m]];
            } else {
              C << p3.x, p3.y, p3.z;
              binary_C = binary_list[pointIdxNKNSearch[n]];
            }
            SemanticTriangularDescriptor single_descriptor;
            single_descriptor.binary_A_ = binary_A;
            single_descriptor.binary_B_ = binary_B;
            single_descriptor.binary_C_ = binary_C;
            single_descriptor.center_ = (A + B + C) / 3;
            single_descriptor.triangle_ << scale * a, scale * b, scale * c;
            single_descriptor.angle_[0] = fabs(5 * normal_1.dot(normal_2));
            single_descriptor.angle_[1] = fabs(5 * normal_1.dot(normal_3));
            single_descriptor.angle_[2] = fabs(5 * normal_3.dot(normal_2));
            single_descriptor.frame_number_ = frame_id;
            // 存储三个顶点的语义标签（按边长顺序：A, B, C）- 第一标签
            // 注意：A, B, C 的语义标签必须与 binary_A, binary_B, binary_C 一一对应
            // 这确保了在 candidate_selector 中语义匹配时，vertex_semantic_[0/1/2] 对应 A/B/C
            // 在 candidate_verify 和 triangle_solver 中，binary_A/B/C.location_ 也对应 A/B/C
            // 如果没有设置语义标签，默认为0（向后兼容）
            single_descriptor.vertex_semantic_ << static_cast<double>(binary_A.semantic_label_),
                                                 static_cast<double>(binary_B.semantic_label_),
                                                 static_cast<double>(binary_C.semantic_label_);
            // 存储三个顶点的语义标签比重（按边长顺序：A, B, C，范围0-1）- 第一标签
            // 如果没有设置语义比例，默认为0（向后兼容）
            single_descriptor.vertex_semantic_ratio_ << binary_A.semantic_ratio_,
                                                        binary_B.semantic_ratio_,
                                                        binary_C.semantic_ratio_;
            // 存储三个顶点的语义标签（按边长顺序：A, B, C）- 第二标签
            single_descriptor.vertex_semantic_2_ << static_cast<double>(binary_A.semantic_label_2_),
                                                    static_cast<double>(binary_B.semantic_label_2_),
                                                    static_cast<double>(binary_C.semantic_label_2_);
            // 存储三个顶点的语义标签比重（按边长顺序：A, B, C，范围0-1）- 第二标签
            single_descriptor.vertex_semantic_ratio_2_ << binary_A.semantic_ratio_2_,
                                                           binary_B.semantic_ratio_2_,
                                                           binary_C.semantic_ratio_2_;
            Eigen::Matrix3d triangle_positon;
            triangle_positon.block<3, 1>(0, 0) = A;
            triangle_positon.block<3, 1>(0, 1) = B;
            triangle_positon.block<3, 1>(0, 2) = C;
            feat_map[position] = true;
            std_list.push_back(single_descriptor);
          }
        }
      }
    }
  }
}

// Note: BtcDescManager::candidate_selector with BTC type is the same as SemanticTriangularDescManager::candidate_selector
// since BTC is a typedef of SemanticTriangularDescriptor. However, the backward compatibility version
// doesn't check semantic labels. We use the semantic version which can handle both cases (with or without semantic labels).

void SemanticTriangularDescManager::candidate_selector(
    const std::vector<SemanticTriangularDescriptor> &current_STD_list,
    std::vector<SemanticTriangularMatchList> &candidate_matcher_vec) {
  int current_frame_id = current_STD_list[0].frame_number_;
  double match_array[20000] = {0};
  std::vector<int> match_list_index;
  std::vector<Eigen::Vector3i> voxel_round;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        Eigen::Vector3i voxel_inc(x, y, z);
        voxel_round.push_back(voxel_inc);
      }
    }
  }
  std::vector<bool> useful_match(current_STD_list.size());
  std::vector<std::vector<size_t>> useful_match_index(current_STD_list.size());
  std::vector<std::vector<SemanticTriangularDescriptor_LOC>> useful_match_position(
      current_STD_list.size());
  std::vector<size_t> index(current_STD_list.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_match[i] = false;
  }
  std::mutex mylock;
  auto t0 = std::chrono::high_resolution_clock::now();

  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) {
        SemanticTriangularDescriptor descriptor = current_STD_list[i];
        SemanticTriangularDescriptor_LOC position;
        double dis_threshold =
            descriptor.triangle_.norm() *
            config_setting_.rough_dis_threshold_;
        for (auto voxel_inc : voxel_round) {
          position.x = (int)(descriptor.triangle_[0] + voxel_inc[0]);
          position.y = (int)(descriptor.triangle_[1] + voxel_inc[1]);
          position.z = (int)(descriptor.triangle_[2] + voxel_inc[2]);
          Eigen::Vector3d voxel_center((double)position.x + 0.5,
                                       (double)position.y + 0.5,
                                       (double)position.z + 0.5);
          if ((descriptor.triangle_ - voxel_center).norm() < 1.5) {
            auto iter = data_base_.find(position);
            if (iter != data_base_.end()) {
              for (size_t j = 0; j < data_base_[position].size(); j++) {
                // 去除帧序列号间隔限制（用于测试匹配性能，允许匹配相邻帧）
                // 原代码：if ((descriptor.frame_number_ - data_base_[position][j].frame_number_) > config_setting_.skip_near_num_)
                // 现在：允许所有帧匹配（包括相邻帧）
                  double dis =
                      (descriptor.triangle_ - data_base_[position][j].triangle_)
                          .norm();
                  if (dis < dis_threshold) {
                    // 描述子相似度：semantic_vertex_match_threshold_!=0 用语义描述子，否则用二进制描述子
                    double similarity;
                    if (config_setting_.semantic_vertex_match_threshold_ != 0) {
                      similarity =
                          (semantic_descriptor_similarity(descriptor.binary_A_, data_base_[position][j].binary_A_, config_setting_.semantic_ratio_threshold_) +
                           semantic_descriptor_similarity(descriptor.binary_B_, data_base_[position][j].binary_B_, config_setting_.semantic_ratio_threshold_) +
                           semantic_descriptor_similarity(descriptor.binary_C_, data_base_[position][j].binary_C_, config_setting_.semantic_ratio_threshold_)) / 3.0;
                    } else {
                      similarity =
                          (binary_similarity(descriptor.binary_A_, data_base_[position][j].binary_A_) +
                           binary_similarity(descriptor.binary_B_, data_base_[position][j].binary_B_) +
                           binary_similarity(descriptor.binary_C_, data_base_[position][j].binary_C_)) / 3.0;
                    }
                    if (similarity > config_setting_.similarity_threshold_) {
                      useful_match[i] = true;
                      useful_match_position[i].push_back(position);
                      useful_match_index[i].push_back(j);
                    } else if (print_debug_info_ && i < 5) {
                      std::cerr << "[candidate_selector] Debug: Similarity too low. "
                                << "similarity=" << similarity
                                << ", threshold=" << config_setting_.similarity_threshold_ << std::endl;
                    }
                  }
                }
              }
            }
          }
      });
  std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>>
      index_recorder;
  auto t1 = std::chrono::high_resolution_clock::now();
  int total_useful_matches = 0;
  for (size_t i = 0; i < useful_match.size(); i++) {
    if (useful_match[i]) {
      total_useful_matches += useful_match_index[i].size();
      for (size_t j = 0; j < useful_match_index[i].size(); j++) {
        match_array[data_base_[useful_match_position[i][j]]
                              [useful_match_index[i][j]]
                                  .frame_number_] += 1;
        Eigen::Vector2i match_index(i, j);
        index_recorder.push_back(match_index);
        match_list_index.push_back(
            data_base_[useful_match_position[i][j]][useful_match_index[i][j]]
                .frame_number_);
      }
    }
  }
  
  if (print_debug_info_) {
    std::cout << "[candidate_selector] Debug: total_useful_matches=" << total_useful_matches 
              << ", index_recorder.size()=" << index_recorder.size() << std::endl;
    // 输出每个帧的投票数（用于诊断）
    std::map<int, int> vote_count_map;
    for (int i = 0; i < 20000; i++) {
      if (match_array[i] > 0) {
        vote_count_map[i] = match_array[i];
      }
    }
    if (!vote_count_map.empty()) {
      std::cout << "[candidate_selector] Debug: Vote counts for frames: ";
      for (const auto& pair : vote_count_map) {
        std::cout << "frame_" << pair.first << "=" << pair.second << " ";
      }
      std::cout << std::endl;
    }
  }

  for (int cnt = 0; cnt < config_setting_.candidate_num_; cnt++) {
    double max_vote = 1;
    int max_vote_index = -1;
    for (int i = 0; i < 20000; i++) {
      if (match_array[i] > max_vote) {
        max_vote = match_array[i];
        max_vote_index = i;
      }
    }
    SemanticTriangularMatchList match_triangle_list;
    // 恢复原版投票阈值，确保位姿估计精度
    if (max_vote_index >= 0 && max_vote >= 5) {
      if (print_debug_info_) {
        std::cout << "[candidate_selector] Debug: Found candidate frame " << max_vote_index 
                  << " with " << max_vote << " votes" << std::endl;
      }
      match_array[max_vote_index] = 0;
      match_triangle_list.match_frame_ = max_vote_index;
      match_triangle_list.match_id_.first = current_frame_id;
      match_triangle_list.match_id_.second = max_vote_index;
      for (size_t i = 0; i < index_recorder.size(); i++) {
        if (match_list_index[i] == max_vote_index) {
          std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor> single_match_pair;
          single_match_pair.first = current_STD_list[index_recorder[i][0]];
          single_match_pair.second =
              data_base_[useful_match_position[index_recorder[i][0]]
                                              [index_recorder[i][1]]]
                        [useful_match_index[index_recorder[i][0]]
                                           [index_recorder[i][1]]];
          match_triangle_list.match_list_.push_back(single_match_pair);
        }
      }
      candidate_matcher_vec.push_back(match_triangle_list);
    } else if (max_vote_index >= 0) {
      // 即使不打印调试信息，也记录投票不足的候选（用于诊断）
      if (print_debug_info_) {
        std::cerr << "[candidate_selector] Debug: Candidate frame " << max_vote_index 
                  << " has only " << max_vote << " votes (need >= 5)" << std::endl;
      }
      // 记录所有投票数，即使不足阈值（用于分析）
      if (max_vote > 0 && max_vote < 5) {
        // 可以在这里添加统计信息，记录哪些帧因为投票不足而失败
      }
    }
  }
  
  if (print_debug_info_) {
    std::cout << "[candidate_selector] Debug: "
              << "current_STD_list.size()=" << current_STD_list.size()
              << ", candidate_matcher_vec.size()=" << candidate_matcher_vec.size() << std::endl;
    if (candidate_matcher_vec.empty()) {
      std::cerr << "[candidate_selector] Warning: No candidates found! "
                << "This may be due to strict semantic matching conditions." << std::endl;
    }
  }
}

// Note: BtcDescManager::candidate_verify with BTCMatchList type is the same as SemanticTriangularDescManager::candidate_verify
// since BTCMatchList is a typedef of SemanticTriangularMatchList. The semantic version implementation above handles both cases.

void SemanticTriangularDescManager::candidate_verify(
    const SemanticTriangularMatchList &candidate_matcher, double &verify_score,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
    std::vector<std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor>> &sucess_match_list) {
  sucess_match_list.clear();
  double dis_threshold = 3;
  std::time_t solve_time = 0;
  std::time_t verify_time = 0;
  int skip_len = (int)(candidate_matcher.match_list_.size() / 50) + 1;
  int use_size = candidate_matcher.match_list_.size() / skip_len;
  std::vector<size_t> index(use_size);
  std::vector<int> vote_list(use_size);
  for (size_t i = 0; i < index.size(); i++) {
    index[i] = i;
  }
  std::mutex mylock;
  auto t0 = std::chrono::high_resolution_clock::now();
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) {
        auto single_pair = candidate_matcher.match_list_[i * skip_len];
        int vote = 0;
        Eigen::Matrix3d test_rot;
        Eigen::Vector3d test_t;
        triangle_solver(single_pair, test_t, test_rot);
        for (size_t j = 0; j < candidate_matcher.match_list_.size(); j++) {
          auto verify_pair = candidate_matcher.match_list_[j];
          Eigen::Vector3d A = verify_pair.first.binary_A_.location_;
          Eigen::Vector3d A_transform = test_rot * A + test_t;
          Eigen::Vector3d B = verify_pair.first.binary_B_.location_;
          Eigen::Vector3d B_transform = test_rot * B + test_t;
          Eigen::Vector3d C = verify_pair.first.binary_C_.location_;
          Eigen::Vector3d C_transform = test_rot * C + test_t;
          double dis_A =
              (A_transform - verify_pair.second.binary_A_.location_).norm();
          double dis_B =
              (B_transform - verify_pair.second.binary_B_.location_).norm();
          double dis_C =
              (C_transform - verify_pair.second.binary_C_.location_).norm();
          if (dis_A < dis_threshold && dis_B < dis_threshold &&
              dis_C < dis_threshold) {
            vote++;
          }
        }
        mylock.lock();
        vote_list[i] = vote;
        mylock.unlock();
      });

  int max_vote_index = 0;
  int max_vote = 0;
  for (size_t i = 0; i < vote_list.size(); i++) {
    if (max_vote < vote_list[i]) {
      max_vote_index = i;
      max_vote = vote_list[i];
    }
  }
  // 恢复原版投票阈值，确保位姿估计精度
  if (max_vote >= 4) {
    auto best_pair = candidate_matcher.match_list_[max_vote_index * skip_len];
    int vote = 0;
    Eigen::Matrix3d best_rot;
    Eigen::Vector3d best_t;
    triangle_solver(best_pair, best_t, best_rot);
    relative_pose.first = best_t;
    relative_pose.second = best_rot;
    for (size_t j = 0; j < candidate_matcher.match_list_.size(); j++) {
      auto verify_pair = candidate_matcher.match_list_[j];
      Eigen::Vector3d A = verify_pair.first.binary_A_.location_;
      Eigen::Vector3d A_transform = best_rot * A + best_t;
      Eigen::Vector3d B = verify_pair.first.binary_B_.location_;
      Eigen::Vector3d B_transform = best_rot * B + best_t;
      Eigen::Vector3d C = verify_pair.first.binary_C_.location_;
      Eigen::Vector3d C_transform = best_rot * C + best_t;
      double dis_A =
          (A_transform - verify_pair.second.binary_A_.location_).norm();
      double dis_B =
          (B_transform - verify_pair.second.binary_B_.location_).norm();
      double dis_C =
          (C_transform - verify_pair.second.binary_C_.location_).norm();
      if (dis_A < dis_threshold && dis_B < dis_threshold &&
          dis_C < dis_threshold) {
        sucess_match_list.push_back(verify_pair);
      }
    }
    // 检查平面点云是否存在且不为空
    // plane_cloud_vec_[0] = target (database), plane_cloud_vec_[1] = source (query)
    // candidate_matcher.match_id_.second 是数据库中的帧ID（应该是0）
    if (plane_cloud_vec_.empty() || 
        candidate_matcher.match_id_.second < 0 || 
        static_cast<size_t>(candidate_matcher.match_id_.second) >= plane_cloud_vec_.size() ||
        plane_cloud_vec_.size() < 2 ||
        plane_cloud_vec_.back()->empty() ||
        plane_cloud_vec_[candidate_matcher.match_id_.second]->empty()) {
      if (print_debug_info_) {
        std::cerr << "[candidate_verify] Warning: plane_cloud is empty or invalid index! "
                  << "plane_cloud_vec_.size()=" << plane_cloud_vec_.size()
                  << ", match_id_.second=" << candidate_matcher.match_id_.second << std::endl;
      }
      verify_score = -1;
    } else {
      verify_score = plane_geometric_verify(
          plane_cloud_vec_.back(),  // source (query) plane cloud at index 1
          plane_cloud_vec_[candidate_matcher.match_id_.second],  // target (database) plane cloud at index 0
          relative_pose);
      
      // Refine pose using ICP optimization (same as original version)
      if (verify_score > 0 && !plane_cloud_vec_.back()->empty() && 
          !plane_cloud_vec_[candidate_matcher.match_id_.second]->empty()) {
        // 获取语义体素映射（如果可用）
        const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> *source_semantic_map = nullptr;
        const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> *target_semantic_map = nullptr;
        
        if (semantic_voxel_map_vec_.size() > 0) {
          // source 是最后一个（当前查询帧）
          size_t source_idx = semantic_voxel_map_vec_.size() - 1;
          if (source_idx < semantic_voxel_map_vec_.size()) {
            source_semantic_map = &semantic_voxel_map_vec_[source_idx];
          }
          
          // target 是候选帧
          size_t target_idx = candidate_matcher.match_id_.second;
          if (target_idx < semantic_voxel_map_vec_.size()) {
            target_semantic_map = &semantic_voxel_map_vec_[target_idx];
          }
        }
        
        PlaneGeomrtricIcp(
            plane_cloud_vec_.back(),  // source (query) plane cloud
            plane_cloud_vec_[candidate_matcher.match_id_.second],  // target (database) plane cloud
            relative_pose,
            source_semantic_map,
            target_semantic_map);
        
        // Re-verify with refined pose
    verify_score = plane_geometric_verify(
        plane_cloud_vec_.back(),
            plane_cloud_vec_[candidate_matcher.match_id_.second],
            relative_pose);
      }
    }
  } else {
    if (print_debug_info_) {
      std::cerr << "[candidate_verify] Debug: max_vote=" << max_vote 
                << " < 2, verify_score set to -1" << std::endl;
    }
    verify_score = -1;
  }
  return;
}

void SemanticTriangularDescManager::triangle_solver(std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor> &std_pair,
                                     Eigen::Vector3d &t, Eigen::Matrix3d &rot) {
  Eigen::Matrix3d src = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
  src.col(0) = std_pair.first.binary_A_.location_ - std_pair.first.center_;
  src.col(1) = std_pair.first.binary_B_.location_ - std_pair.first.center_;
  src.col(2) = std_pair.first.binary_C_.location_ - std_pair.first.center_;
  ref.col(0) = std_pair.second.binary_A_.location_ - std_pair.second.center_;
  ref.col(1) = std_pair.second.binary_B_.location_ - std_pair.second.center_;
  ref.col(2) = std_pair.second.binary_C_.location_ - std_pair.second.center_;
  Eigen::Matrix3d covariance = src * ref.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d U = svd.matrixU();
  rot = V * U.transpose();
  if (rot.determinant() < 0) {
    Eigen::Matrix3d K;
    K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
    rot = V * K * U.transpose();
  }
  t = -rot * std_pair.first.center_ + std_pair.second.center_;
}

// Note: BtcDescManager::triangle_solver with BTC type is the same as SemanticTriangularDescManager::triangle_solver
// since BTC is a typedef of SemanticTriangularDescriptor. The semantic version implementation above handles both cases.

double BtcDescManager::plane_geometric_verify(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform) {
  // 检查点云是否为空
  if (source_cloud->empty() || target_cloud->empty()) {
    if (print_debug_info_) {
      std::cerr << "[plane_geometric_verify] Warning: source_cloud or target_cloud is empty!" << std::endl;
    }
    return 0.0;
  }
  
  Eigen::Vector3d t = transform.first;
  Eigen::Matrix3d rot = transform.second;
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < target_cloud->size(); i++) {
    pcl::PointXYZ pi;
    pi.x = target_cloud->points[i].x;
    pi.y = target_cloud->points[i].y;
    pi.z = target_cloud->points[i].z;
    input_cloud->push_back(pi);
  }

  kd_tree->setInputCloud(input_cloud);
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  double useful_match = 0;
  double normal_threshold = config_setting_.normal_threshold_;
  double dis_threshold = config_setting_.dis_threshold_;
  for (size_t i = 0; i < source_cloud->size(); i++) {
    pcl::PointXYZINormal searchPoint = source_cloud->points[i];
    pcl::PointXYZ use_search_point;
    use_search_point.x = searchPoint.x;
    use_search_point.y = searchPoint.y;
    use_search_point.z = searchPoint.z;
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    use_search_point.x = pi[0];
    use_search_point.y = pi[1];
    use_search_point.z = pi[2];
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y,
                       searchPoint.normal_z);
    ni = rot * ni;
    if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      pcl::PointXYZINormal nearstPoint =
          target_cloud->points[pointIdxNKNSearch[0]];
      Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
      Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y,
                          nearstPoint.normal_z);
      Eigen::Vector3d normal_inc = ni - tni;
      Eigen::Vector3d normal_add = ni + tni;
      double point_to_plane = fabs(tni.transpose() * (pi - tpi));
      if ((normal_inc.norm() < normal_threshold ||
           normal_add.norm() < normal_threshold) &&
          point_to_plane < dis_threshold) {
        useful_match++;
      }
    }
  }
  return useful_match / source_cloud->size();
}


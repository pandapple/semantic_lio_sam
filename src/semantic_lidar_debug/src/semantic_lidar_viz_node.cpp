/**
 * Debug node: subscribe lidar PointCloud2, load matching SemanticKITTI .label from folder,
 * publish PointXYZRGB with semantic colors for RViz (frame-by-frame in bag order).
 */
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

namespace velodyne_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  std::uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(
    velodyne_ros::Point,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, time, time)(std::uint16_t, ring, ring))

static std::array<uint8_t, 3> labelToRgb(uint32_t label) {
  static const std::unordered_map<uint32_t, std::array<uint8_t, 3>> semantic_colors = {
      {0, {124, 176, 138}},   {1, {124, 176, 138}},   {10, {255, 0, 0}},      {11, {255, 128, 0}},
      {13, {255, 0, 128}},    {15, {255, 255, 0}},    {16, {0, 200, 0}},      {18, {0, 255, 255}},
      {30, {255, 80, 80}},    {31, {255, 128, 255}},  {32, {128, 0, 255}},    {40, {255, 255, 0}},
      {44, {160, 32, 240}},   {48, {0, 128, 255}},    {49, {255, 165, 0}},    {50, {0, 0, 255}},
      {51, {128, 255, 255}},  {52, {0, 64, 255}},     {60, {0, 255, 128}},    {70, {0, 255, 0}},
      {71, {139, 69, 19}},    {72, {205, 133, 63}},   {80, {255, 0, 255}},    {81, {192, 192, 192}},
      {99, {255, 255, 255}}};
  auto it = semantic_colors.find(label & 0xFFFFu);
  if (it != semantic_colors.end()) return it->second;
  return semantic_colors.at(0);
}

// 與 merge_semantic_pt.py 一致：labels = np.fromfile(..., dtype=np.uint32); semantic = labels & 0xFFFF
static bool loadLabels(const std::string& path, std::vector<uint32_t>& labels) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.seekg(0, std::ios::end);
  const auto sz = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  const size_t n = sz / sizeof(uint32_t);
  labels.resize(n);
  for (size_t i = 0; i < n; ++i) {
    uint32_t v;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    labels[i] = v & 0xFFFFu;
  }
  return true;
}

// 與 merge_semantic_pt.py 一致：scan = np.fromfile(bin, dtype=np.float32).reshape(-1, 4); points = scan[:, :3]
static bool loadKittiVelodyneBin(const std::string& path, std::vector<std::array<float, 3>>& out_xyz) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.seekg(0, std::ios::end);
  const auto bytes = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  if (bytes % (sizeof(float) * 4) != 0) return false;
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

int main(int argc, char** argv) {
  ros::init(argc, argv, "semantic_lidar_viz_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  std::string lidar_topic = "/points_raw";
  std::string label_folder;
  std::string velodyne_folder;  // 若非空：嚴格按 merge_semantic_pt.py 用同序的 .bin + .label（不依賴 ROS 點序）
  std::string output_topic = "/semantic_debug_cloud";
  int label_index_offset = 0;

  pnh.param<std::string>("lidar_topic", lidar_topic, lidar_topic);
  pnh.param<std::string>("label_folder", label_folder, label_folder);
  pnh.param<std::string>("velodyne_folder", velodyne_folder, velodyne_folder);
  pnh.param<std::string>("output_topic", output_topic, output_topic);
  pnh.param("label_index_offset", label_index_offset, 0);

  if (label_folder.empty()) {
    ROS_FATAL("~label_folder is empty. Set e.g. /semantic_lidar_viz/label_folder");
    return 1;
  }
  if (!velodyne_folder.empty()) {
    ROS_INFO("Using velodyne_folder + label_folder (merge_semantic_pt.py row alignment); ROS cloud is sync-only.");
  } else {
    ROS_WARN("~velodyne_folder is empty: coloring uses ROS point order vs .label row order — often WRONG for bags. "
             "Set velodyne_folder to <sequence>/velodyne for KITTI-aligned semantics.");
  }

  ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>(output_topic, 2);

  int frame_idx = 0;

  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>(
      lidar_topic, 10,
      [&](const sensor_msgs::PointCloud2::ConstPtr& msg) {
        pcl::PointCloud<velodyne_ros::Point> cloud;
        if (velodyne_folder.empty()) {
          try {
            pcl::fromROSMsg(*msg, cloud);
          } catch (const std::exception& e) {
            ROS_WARN_THROTTLE(2.0, "pcl::fromROSMsg failed: %s", e.what());
            return;
          }
        }

        const int label_file_idx = frame_idx + label_index_offset;
        if (label_file_idx < 0) {
          ROS_WARN_THROTTLE(1.0, "label_file_idx=%d < 0, skip", label_file_idx);
          frame_idx++;
          return;
        }

        std::stringstream ss_lab;
        ss_lab << label_folder;
        if (label_folder.back() != '/') ss_lab << '/';
        ss_lab << std::setfill('0') << std::setw(6) << label_file_idx << ".label";

        std::vector<uint32_t> labels;
        if (!loadLabels(ss_lab.str(), labels)) {
          ROS_WARN_THROTTLE(1.0, "Cannot open label: %s", ss_lab.str().c_str());
          frame_idx++;
          return;
        }

        pcl::PointCloud<pcl::PointXYZRGB> out;

        if (!velodyne_folder.empty()) {
          std::stringstream ss_bin;
          ss_bin << velodyne_folder;
          if (velodyne_folder.back() != '/') ss_bin << '/';
          ss_bin << std::setfill('0') << std::setw(6) << label_file_idx << ".bin";
          std::vector<std::array<float, 3>> bin_xyz;
          if (!loadKittiVelodyneBin(ss_bin.str(), bin_xyz)) {
            ROS_WARN_THROTTLE(1.0, "Cannot open velodyne bin: %s", ss_bin.str().c_str());
            frame_idx++;
            return;
          }
          if (bin_xyz.size() != labels.size()) {
            ROS_WARN_THROTTLE(2.0, "frame %d: bin=%zu label=%zu (%s vs %s)", frame_idx, bin_xyz.size(),
                              labels.size(), ss_bin.str().c_str(), ss_lab.str().c_str());
            frame_idx++;
            return;
          }
          const size_t n = bin_xyz.size();
          out.resize(n);
          for (size_t i = 0; i < n; ++i) {
            auto rgb = labelToRgb(labels[i]);
            out.points[i].x = bin_xyz[i][0];
            out.points[i].y = bin_xyz[i][1];
            out.points[i].z = bin_xyz[i][2];
            out.points[i].r = rgb[0];
            out.points[i].g = rgb[1];
            out.points[i].b = rgb[2];
          }
        } else {
          const size_t n_pts = cloud.size();
          const size_t n_lab = labels.size();
          const size_t n = std::min(n_pts, n_lab);
          if (n_pts != n_lab) {
            ROS_WARN_THROTTLE(2.0,
                              "frame %d: cloud=%zu label=%zu (use min=%zu) file=%s",
                              frame_idx, n_pts, n_lab, n, ss_lab.str().c_str());
          }
          out.resize(n);
          for (size_t i = 0; i < n; ++i) {
            const auto& p = cloud.points[i];
            auto rgb = labelToRgb(labels[i]);
            out.points[i].x = p.x;
            out.points[i].y = p.y;
            out.points[i].z = p.z;
            out.points[i].r = rgb[0];
            out.points[i].g = rgb[1];
            out.points[i].b = rgb[2];
          }
        }

        sensor_msgs::PointCloud2 out_msg;
        pcl::toROSMsg(out, out_msg);
        out_msg.header = msg->header;

        pub.publish(out_msg);

        if (frame_idx < 5 || frame_idx % 50 == 0) {
          ROS_INFO("frame %d -> %s colored %zu pts", frame_idx, ss_lab.str().c_str(), out.size());
        }
        frame_idx++;
      });

  ROS_INFO("semantic_lidar_viz: sub=%s pub=%s label_folder=%s offset=%d",
           lidar_topic.c_str(), output_topic.c_str(), label_folder.c_str(), label_index_offset);

  ros::spin();
  return 0;
}

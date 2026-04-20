#!/usr/bin/env python3
"""
KITTI velodyne -> rosbag。点云消息中点的顺序与对应 .bin 文件内 float32 行顺序一致，
以便与 SemanticKITTI .label（第 i 行对应 .bin 第 i 个点）一一对应。

要点：
- 仅收集 *.bin，并按帧号数值排序（非纯字典序）。
- 写 PointCloud2 时用显式 list of tuples 按行 0..N-1 打包，与 np.fromfile().reshape(-1,4) 行序一致。
"""
import os
import rospy
import rosbag
import numpy as np
from datetime import datetime
from sensor_msgs.msg import Imu, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pcl2

# ========= 时间解析 =========
def load_timestamps(ts_file):
    with open(ts_file) as f:
        times = [
            datetime.strptime(l.strip()[:-4], '%Y-%m-%d %H:%M:%S.%f')
            for l in f
        ]
    return times


# ========= IMU =========
def save_imu(bag, imu_path, frame_id="imu"):
    print("Exporting IMU...")

    ts_file = os.path.join(imu_path, "timestamps.txt")
    data_dir = os.path.join(imu_path, "data")

    times = load_timestamps(ts_file)
    files = sorted(os.listdir(data_dir))

    assert len(times) == len(files), "IMU 时间戳和数据数量不一致！"

    for t, f_name in zip(times, files):
        with open(os.path.join(data_dir, f_name)) as f:
            data = list(map(float, f.readline().split()))

        imu = Imu()
        imu.header = Header()
        imu.header.stamp = rospy.Time.from_sec(float(t.strftime("%s.%f")))
        imu.header.frame_id = frame_id

        # KITTI OXTS字段
        imu.linear_acceleration.x = data[11]
        imu.linear_acceleration.y = data[12]
        imu.linear_acceleration.z = data[13]

        imu.angular_velocity.x = data[17]
        imu.angular_velocity.y = data[18]
        imu.angular_velocity.z = data[19]

        bag.write("/imu/data", imu, imu.header.stamp)


def _sorted_velodyne_bin_files(velo_path):
    """仅 *.bin，按帧编号数值排序（与 KITTI / SemanticKITTI 帧序一致）。"""
    names = [
        f
        for f in os.listdir(velo_path)
        if f.endswith(".bin") and not f.startswith(".")
    ]

    def frame_key(name):
        stem = os.path.splitext(name)[0]
        try:
            return int(stem)
        except ValueError:
            return stem

    return sorted(names, key=frame_key)


def _bin_to_pointcloud2_tuples(scan_xyzi):
    """
    scan_xyzi: (N,4) float32，与 merge_semantic_pt.py 中 scan 一致。
    返回长度为 N 的 list，第 i 个元素为 .bin 第 i 行的 (x,y,z,intensity)，
    供 pcl2.create_cloud 按序写入 buffer（与 label 下标 i 对齐）。
    """
    # tolist() 按 C 行序展开，与 reshape(-1,4) 行顺序一致；tuple(map(float,...)) 供 struct 打包
    return [tuple(map(float, row)) for row in scan_xyzi.tolist()]


# ========= LiDAR =========
def save_velodyne(bag, velo_path, ts_path, topic="/velodyne_points", frame_id="velodyne"):
    print("Exporting Velodyne...")

    files = _sorted_velodyne_bin_files(velo_path)
    times = load_timestamps(ts_path)

    # ===== 关键修改 =====
    if len(times) < len(files):
        raise RuntimeError("时间戳数量少于点云数量！")

    if len(times) > len(files):
        print(f"[WARN] timestamps 多于点云：{len(times)} vs {len(files)}，自动截断")
        times = times[:len(files)]
    # ===================

    for t, f_name in zip(times, files):
        file_path = os.path.join(velo_path, f_name)

        raw = np.fromfile(file_path, dtype=np.float32)
        if raw.size % 4 != 0:
            raise ValueError(f"Invalid velodyne bin size (not multiple of 4 floats): {file_path}")
        scan = raw.reshape(-1, 4)

        header = Header()
        header.stamp = rospy.Time.from_sec(float(t.strftime("%s.%f")))
        header.frame_id = frame_id

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]

        # 与 .bin 行顺序严格一致，对应 SemanticKITTI .label 的第 i 个 uint32
        points_ordered = _bin_to_pointcloud2_tuples(scan)
        cloud = pcl2.create_cloud(header, fields, points_ordered)
        bag.write(topic, cloud, header.stamp)


# ========= MAIN =========
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--velo", required=True, help="velodyne folder path")
    parser.add_argument("--velo_ts", required=True, help="velodyne timestamps.txt path")
    parser.add_argument("--imu", required=True, help="oxts folder path")
    parser.add_argument("--out", default="lidar_imu.bag")
    parser.add_argument(
        "--lidar_topic",
        default="/velodyne_points",
        help="bag 中点云 topic（需与回放时 remap 一致；顺序已与 .bin 一致）",
    )

    args = parser.parse_args()

    rospy.init_node("kitti_to_bag")

    bag = rosbag.Bag(args.out, 'w')

    try:
        save_imu(bag, args.imu)
        save_velodyne(bag, args.velo, args.velo_ts, topic=args.lidar_topic)
    finally:
        bag.close()
        print("Done!")

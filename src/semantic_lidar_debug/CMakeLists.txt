cmake_minimum_required(VERSION 3.0.2)
project(semantic_lidar_debug)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
)

find_package(PCL REQUIRED COMPONENTS common io)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs pcl_ros pcl_conversions
  DEPENDS PCL
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(semantic_lidar_viz_node src/semantic_lidar_viz_node.cpp)
target_link_libraries(semantic_lidar_viz_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})

install(TARGETS semantic_lidar_viz_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch rviz DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

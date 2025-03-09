cmake_minimum_required(VERSION 3.5)
project(object_tracker)

# C++ 표준 (필요시 14 이상으로 설정)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# ament_cmake 및 의존 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(rclpy REQUIRED) 

# tracker.py node
install(PROGRAMS
  src/tracker.py  # Python 스크립트 경로
  DESTINATION lib/${PROJECT_NAME}  # ROS 2 실행 파일들이 위치하는 기본 경로
)

# launch 파일이나 기타 리소스 디렉토리도 설치할 경우
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

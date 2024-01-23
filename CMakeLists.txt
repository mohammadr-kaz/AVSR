cmake_minimum_required(VERSION 3.0.2)
project(docking)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)
find_package(gazebo REQUIRED)

catkin_package(

)

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_library(earth_orbit SHARED plugin/OrbitWorldPlugin.cpp)
target_link_libraries(earth_orbit ${GAZEBO_LIBRARIES})

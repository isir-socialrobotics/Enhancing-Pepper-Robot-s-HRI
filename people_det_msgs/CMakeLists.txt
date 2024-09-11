cmake_minimum_required(VERSION 2.8.3)
project(people_det_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
	std_msgs
	sensor_msgs
	geometry_msgs
	message_generation
	rospy
)

## Generate messages in the 'msg' folder
add_message_files(
   FILES
    HumanBodyJoints.msg
    HumanBodyJointsArray.msg
    HumanDetectionFov.msg
    HumanDetectionFovArray.msg
    Joint2D.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
## CATKIN_DEPENDS: catkin_packages dependent projects also need
catkin_package(CATKIN_DEPENDS 
					message_runtime 
					std_msgs
					sensor_msgs
					rospy
					)

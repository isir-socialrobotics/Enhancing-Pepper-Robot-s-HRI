# ROS message for a detected person with their body orientation and head orientation

std_msgs/Header header
uint32 person_ID

PersonDetection body_joints

geometry_msgs/Quaternion body_orientation

geometry_msgs/Quaternion head_orientation

geometry_msgs/Point position
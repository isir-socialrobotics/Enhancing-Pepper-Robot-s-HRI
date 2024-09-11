import os
import cv2
import numpy as np
from ultralytics import YOLO

import rospy
from sensor_msgs.msg import Image

from people_det_msgs.msg import HumanBodyJointsArray, HumanBodyJoints, Joint2D


class YoloObjectDetection:
    def __init__(self):
        self.yolo = YOLO("yolov8n-pose.pt")

        self.keypoint_labels = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        self.image_sub = rospy.Subscriber("/head_mount_kinect2/rgb/image_raw", Image,
            self.image_callback, queue_size=1)

        self.keypoints_pub = rospy.Publisher("/vision/human_body_joints", HumanBodyJointsArray, queue_size=1)

    def image_callback(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        # sample_image_file = "/home/javad/workspace/camm_catkin_ws/src/CAMM/CAMM/assets/gazebo_sample_01.jpg"
        # img = cv2.imread(sample_image_file)

        results = self.yolo(img, show=False)

        # publish joint keypoints
        body_joints_array = HumanBodyJointsArray()
        for det in results[0].keypoints:
            body_joints = HumanBodyJoints()
            det = det.data.cpu().numpy().squeeze()
            body_joints.nose = Joint2D(x=det[0][0], y=det[0][1], visible=det[0][2])
            body_joints.left_eye = Joint2D(x=det[1][0], y=det[1][1], visible=det[1][2])
            body_joints.right_eye = Joint2D(x=det[2][0], y=det[2][1], visible=det[2][2])
            body_joints.left_ear = Joint2D(x=det[3][0], y=det[3][1], visible=det[3][2])
            body_joints.right_ear = Joint2D(x=det[4][0], y=det[4][1], visible=det[4][2])
            body_joints.left_shoulder = Joint2D(x=det[5][0], y=det[5][1], visible=det[5][2])
            body_joints.right_shoulder = Joint2D(x=det[6][0], y=det[6][1], visible=det[6][2])
            body_joints.left_elbow = Joint2D(x=det[7][0], y=det[7][1], visible=det[7][2])
            body_joints.right_elbow = Joint2D(x=det[8][0], y=det[8][1], visible=det[8][2])
            body_joints.left_wrist = Joint2D(x=det[9][0], y=det[9][1], visible=det[9][2])
            body_joints.right_wrist = Joint2D(x=det[10][0], y=det[10][1], visible=det[10][2])
            body_joints.left_hip = Joint2D(x=det[11][0], y=det[11][1], visible=det[11][2])
            body_joints.right_hip = Joint2D(x=det[12][0], y=det[12][1], visible=det[12][2])
            body_joints.left_knee = Joint2D(x=det[13][0], y=det[13][1], visible=det[13][2])
            body_joints.right_knee = Joint2D(x=det[14][0], y=det[14][1], visible=det[14][2])
            body_joints.left_ankle = Joint2D(x=det[15][0], y=det[15][1], visible=det[15][2])
            body_joints.right_ankle = Joint2D(x=det[16][0], y=det[16][1], visible=det[16][2])

            body_joints_array.detections.append(body_joints)

        self.keypoints_pub.publish(body_joints_array)


if __name__ == "__main__":
    rospy.init_node("yolo_object_detection_node")
    YoloObjectDetection()
    rospy.spin()
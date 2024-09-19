#!/usr/bin/env python3

import cv2
import message_filters
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from ultralytics import YOLO

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseArray, Vector3

from people_det_3d.kalman import KalmanFilter1D
from people_det_3d.utils import calculate_3d


class PeopleDetection3D:
    def __init__(self, enable_visualization=True):
        # Initialize ROS
        rospy.init_node('people_detection_3d', anonymous=True)

        # Publishers
        self.keypoints_pub = rospy.Publisher('/keypoints', PoseArray, queue_size=10)
        self.arrow_body_pub = rospy.Publisher('/arrow_body_vector', Vector3, queue_size=10)
        self.arrow_gaze_pub = rospy.Publisher('/arrow_gaze_vector', Vector3, queue_size=10)
        self.turned_man_pub = rospy.Publisher('/turned_man', String, queue_size=10)

        # Kalman filters
        self.kf_position = KalmanFilter1D(0.0, 1.0, 0.1, 1.0)
        self.kf_gaze = KalmanFilter1D(0.0, 1.0, 0.1, 1.0)

        # Subscribe to RealSense camera topics using message_filters for synchronization
        image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
        depth_info_sub = message_filters.Subscriber('/camera/depth/camera_info', CameraInfo)

        # Synchronize the incoming data streams
        sync_sub = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub, depth_info_sub], 10, 0.1)
        sync_sub.registerCallback(self.detect)

        self.person_id_counter = -1
        self.yolo_model = YOLO('yolov8n-pose')

        self.enable_visualization = enable_visualization
        if self.enable_visualization:
            import matplotlib
            matplotlib.use('TkAgg')
            self.initialize_visualization()

    def initialize_visualization(self):
        import matplotlib.pyplot as plt
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(0, 4)
        self.ax.set_zlim(-1, 3)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.cube_vertices = np.array([[-0.05, -0.05, -0.05],
                                       [0.05, -0.05, -0.05],
                                       [0.05, 0.05, -0.05],
                                       [-0.05, 0.05, -0.05],
                                       [-0.05, -0.05, 0.05],
                                       [0.05, -0.05, 0.05],
                                       [0.05, 0.05, 0.05],
                                       [-0.05, 0.05, 0.05]])
        self.cube_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                           (4, 5), (5, 6), (6, 7), (7, 4),
                           (0, 4), (1, 5), (2, 6), (3, 7)]

    def visualize(self, keypoints_3d):
        import matplotlib.pyplot as plt
        if not self.enable_visualization:
            return
        self.ax.clear()
        for edge in self.cube_edges:
            start, end = edge
            self.ax.plot([self.cube_vertices[start][0], self.cube_vertices[end][0]],
                         [self.cube_vertices[start][1], self.cube_vertices[end][1]],
                         [self.cube_vertices[start][2], self.cube_vertices[end][2]], 'k')
        for label, x, y, z in keypoints_3d:
            self.ax.scatter(x, y, z)
        plt.draw()
        plt.pause(0.001)

    def detect(self, color_image, depth_image, depth_cam_info):
        """Callback method to handle synchronized image and depth data."""
        # Convert ROS Image message to numpy array
        color_image_np = self.ros_img_to_np(color_image)
        depth_image_np = self.ros_img_to_np(depth_image)

        # Detect people using YOLO model
        persons = self.yolo_model(color_image_np)

        all_keypoints_3d = []
        for results in persons:
            for result in results:
                self.person_id_counter += 1
                person_id = self.person_id_counter

                if hasattr(result, 'keypoints'):
                    keypoints_3d = self.process_keypoints(result, depth_image_np, depth_cam_info)
                    all_keypoints_3d.extend(keypoints_3d)

                    self.publish_keypoints(keypoints_3d)
                    self.publish_vectors()

        if self.enable_visualization:
            self.visualize(all_keypoints_3d)

    def process_keypoints(self, result, depth_image, depth_cam_info):
        """Convert detected 2D keypoints to 3D using depth information."""
        kpts = result.keypoints.xy.cpu().numpy()
        keypoints_list = kpts.flatten().tolist()
        keypoints_3d = []

        for x, y in zip(keypoints_list[::2], keypoints_list[1::2]):
            x_3d, y_3d, z_3d, _ = calculate_3d(int(x), int(y), depth_image, depth_cam_info,
                                               depth_image.shape[1], depth_image.shape[0])
            if not np.isnan(z_3d) and (x_3d != 0 or y_3d != 0 or z_3d != 0):
                keypoints_3d.append(('Keypoint', x_3d, y_3d, z_3d))

        return keypoints_3d

    def publish_keypoints(self, keypoints_3d):
        """Publish the detected 3D keypoints to a ROS topic."""
        keypoints_poses = PoseArray()
        for _, x_3d, y_3d, z_3d in keypoints_3d:
            pose = Pose()
            pose.position.x = x_3d
            pose.position.y = y_3d
            pose.position.z = z_3d
            keypoints_poses.poses.append(pose)
        self.keypoints_pub.publish(keypoints_poses)

    def publish_vectors(self):
        """Publish the body and gaze vectors to the corresponding ROS topics."""
        arrow_body_vector = Vector3()
        arrow_gaze_vector = Vector3()
        # Populate with real values in practice
        self.arrow_body_pub.publish(arrow_body_vector)
        self.arrow_gaze_pub.publish(arrow_gaze_vector)

    def detect_people_loop(self):
        """Main loop for detecting people."""
        rospy.spin()  # ROS will call the callbacks as messages arrive

    def ros_img_to_np(self, ros_img):
        """Convert ROS Image message to numpy array."""
        try:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            return bridge.imgmsg_to_cv2(ros_img, desired_encoding='passthrough')
        except Exception as e:
            rospy.logerr("Error converting ROS image to numpy array: %s", str(e))
            return None

    def run(self):
        """Run the main detection loop."""
        try:
            self.detect_people_loop()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = PeopleDetection3D(enable_visualization=True)
    detector.run()

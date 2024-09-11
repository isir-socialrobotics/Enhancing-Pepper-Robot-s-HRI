#!/usr/bin/env python3

import cv2
import numpy as np
import pyrealsense2 as rs
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

        self.keypoints_pub = rospy.Publisher('/keypoints', PoseArray, queue_size=10)
        self.arrow_body_pub = rospy.Publisher('/arrow_body_vector', Vector3, queue_size=10)
        self.arrow_gaze_pub = rospy.Publisher('/arrow_gaze_vector', Vector3, queue_size=10)
        self.turned_man_pub = rospy.Publisher('/turned_man', String, queue_size=10)

        # Kalman filters
        self.kf_position = KalmanFilter1D(0.0, 1.0, 0.1, 1.0)
        self.kf_gaze = KalmanFilter1D(0.0, 1.0, 0.1, 1.0)


        self.person_id_counter = -1
        self.yolo_model = YOLO('yolov8n-pose')

        # subscribe to RealSense camera

        # Realsense camera
        # self.pipeline, self.config, self.align, self.color_intrinsics = self.initialize_realsense()

        self.enable_visualization = enable_visualization
        if self.enable_visualization:
            import matplotlib
            matplotlib.use('TkAgg')
            self.initialize_visualization()

    def initialize_realsense(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        align_to = rs.stream.color
        align = rs.align(align_to)
        color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return pipeline, config, align, color_intrinsics

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
        
    def detect_people_loop(self):
        # Buffers to store the last azimuth values
        azimuth_buffer = []
        gaze_azimuth_buffer = []
    
        while not rospy.is_shutdown():
            # Acquire frames from the RealSense camera
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
    
            if not aligned_depth_frame or not color_frame:
                continue
    
            # Convert frames to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
    
            # Detect people using the YOLO model
            persons = self.yolo_model(color_image)
    
            all_keypoints_3d = []
    
            for results in persons:
                for result in results:
                    if hasattr(result, 'keypoints'):
                        # Extract 2D keypoints and convert them to 3D
                        kpts = result.keypoints.xy.cpu().numpy()
                        keypoints_list = kpts.flatten().tolist()
                        labels = [self.index_to_label.get(i, '') for i in range(len(keypoints_list) // 2)]
    
                        keypoints_2d = {}
                        keypoints_3d = []
    
                        # Process keypoints to obtain 3D coordinates
                        for i, (x, y) in enumerate(zip(keypoints_list[::2], keypoints_list[1::2])):
                            x_3d, y_3d, z_3d, min_depth = calculate_3d(int(x), int(y), aligned_depth_frame, self.color_intrinsics, depth_image.shape[1], depth_image.shape[0])
                            if not np.isnan(z_3d) and (x_3d != 0 or y_3d != 0 or z_3d != 0):
                                keypoints_3d.append((labels[i], x_3d, y_3d, z_3d))
                                keypoints_2d[labels[i]] = (int(x), int(y))
    
                        # Calculate body and gaze orientation
                        if 'Shoulder.L' in keypoints_2d and 'Shoulder.R' in keypoints_2d and 'Pelvis' in keypoints_2d and 'Neck' in keypoints_2d:
                            shoulder_l = next(kp for kp in keypoints_3d if kp[0] == 'Shoulder.L')
                            shoulder_r = next(kp for kp in keypoints_3d if kp[0] == 'Shoulder.R')
                            pelvis = next(kp for kp in keypoints_3d if kp[0] == 'Pelvis')
                            neck = next(kp for kp in keypoints_3d if kp[0] == 'Neck')
    
                            p1, p2, p3 = np.array(shoulder_l[1:]), np.array(shoulder_r[1:]), np.array(pelvis[1:])
                            arrow_start, arrow_end, _, _ = calculate_plane_and_arrow(p1, p2, p3, p3, np.array(neck[1:]), arrow_length=3)
    
                            azimuth = calculate_azimuth(arrow_end - arrow_start, pelvis[1:])
                            azimuth_buffer.append(azimuth)
                            if len(azimuth_buffer) > 5:
                                azimuth_buffer.pop(0)
    
                            if should_use_kalman(azimuth):
                                for previous_azimuth in azimuth_buffer:
                                    self.kf_position.update(previous_azimuth)
                                self.kf_position.predict()
                                predicted_azimuth = self.kf_position.get_state()
    
                            # Calculate and manage gaze azimuth
                            if 'Eye.L' in keypoints_2d and 'Eye.R' in keypoints_2d:
                                eye_l = next(kp for kp in keypoints_3d if kp[0] == 'Eye.L')
                                eye_r = next(kp for kp in keypoints_3d if kp[0] == 'Eye.R')
    
                                p4, p5 = np.array(eye_l[1:]), np.array(eye_r[1:])
                                gaze_arrow_start, gaze_arrow_end, _, _ = calculate_plane_and_arrow(p4, p5, np.array(neck[1:]), np.array(neck[1:]), np.array(neck[1:]), arrow_length=3)
    
                                gaze_azimuth = calculate_azimuth_gaze(gaze_arrow_end - gaze_arrow_start, neck[1:])
                                gaze_azimuth_buffer.append(gaze_azimuth)
                                if len(gaze_azimuth_buffer) > 5:
                                    gaze_azimuth_buffer.pop(0)
    
                                if should_use_kalman_gaze(gaze_azimuth):
                                    for previous_azimuth in gaze_azimuth_buffer:
                                        self.kf_gaze.update(previous_azimuth)
                                    self.kf_gaze.predict()
                                    predicted_gaze_azimuth = self.kf_gaze.get_state()
    
                        # Visualization and publishing of results
                        self.visualize(all_keypoints_3d)
                        self.publish_keypoints(all_keypoints_3d)
                        self.publish_vectors()
    
            cv2.imshow('YOLO Keypoints', color_image)
            if cv2.waitKey(1) == ord('q'):
                break



    def detect(self, color_image):
        persons = self.yolo_model(color_image)

        all_keypoints_3d = []
        for results in persons:
            for result in results:
                self.person_id_counter += 1
                person_id = self.person_id_counter

                if hasattr(result, 'keypoints'):
                    keypoints_3d = self.process_keypoints(result, aligned_depth_frame, depth_image)
                    all_keypoints_3d.extend(keypoints_3d)

                    self.publish_keypoints(keypoints_3d)
                    self.publish_vectors()

        if self.enable_visualization:
            self.visualize(all_keypoints_3d)


    def process_keypoints(self, result, aligned_depth_frame, depth_image):
        kpts = result.keypoints.xy.cpu().numpy()
        keypoints_list = kpts.flatten().tolist()
        keypoints_3d = []

        for x, y in zip(keypoints_list[::2], keypoints_list[1::2]):
            x_3d, y_3d, z_3d, _ = calculate_3d(int(x), int(y), aligned_depth_frame, self.color_intrinsics,
                                               depth_image.shape[1], depth_image.shape[0])
            if not np.isnan(z_3d) and (x_3d != 0 or y_3d != 0 or z_3d != 0):
                keypoints_3d.append(('Keypoint', x_3d, y_3d, z_3d))

        return keypoints_3d

    def publish_keypoints(self, keypoints_3d):
        keypoints_poses = PoseArray()
        for _, x_3d, y_3d, z_3d in keypoints_3d:
            pose = Pose()
            pose.position.x = x_3d
            pose.position.y = y_3d
            pose.position.z = z_3d
            keypoints_poses.poses.append(pose)
        self.keypoints_pub.publish(keypoints_poses)

    def publish_vectors(self):
        arrow_body_vector = Vector3()
        arrow_gaze_vector = Vector3()
        # Populate with real values in practice
        self.arrow_body_pub.publish(arrow_body_vector)
        self.arrow_gaze_pub.publish(arrow_gaze_vector)

    def run(self):
        try:
            self.detect_people_loop()
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = PeopleDetection3D(enable_visualization=True)
    detector.run()

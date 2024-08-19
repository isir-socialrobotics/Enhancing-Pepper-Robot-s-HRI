import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R
import cv2.aruco as aruco
import itertools

def calculate_3d(x, y, depth_image, intrinsics, window_size_small=60, window_size_large=100, max_depth_threshold=4000):
    if x == 0 and y == 0:
        return 0.0, 0.0, 0.0, 0.0

    min_depth = float('inf')

    for window_size in [window_size_small, window_size_large]:
        for i in range(-window_size // 2, window_size // 2 + 1):
            for j in range(-window_size // 2, window_size // 2 + 1):
                x_pixel = int(x) + i
                y_pixel = int(y) + j

                if 0 <= x_pixel < depth_image.shape[1] and 0 <= y_pixel < depth_image.shape[0]:
                    depth = depth_image[y_pixel, x_pixel] / 1000.0  # Divide by 1000 to convert to meters
                    if depth != 0 and depth < min_depth:
                        min_depth = depth
        
        # Check the depth value after the first window
        if min_depth <= max_depth_threshold / 1000.0:  # Convert mm to meters
            break

    if min_depth == float('inf'):
        return 0.0, 0.0, 0.0, 0.0

    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], min_depth)

    x_3d, z_3d, y_3d = point

    return x_3d, y_3d, -z_3d, min_depth

def calculate_plane_and_arrow(p1, p2, p3, p4, p5, arrow_length):
    v1 = np.subtract(p2, p1)
    v2 = np.subtract(p3, p1)

    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    normal_xy = np.array([normal[0], normal[1], 0])
    normal_xy = normal_xy / np.linalg.norm(normal_xy)

    arrow_start = p4
    arrow_end = p4 + arrow_length * normal_xy

    return arrow_start, arrow_end, normal_xy

def euler_from_vectors(v_from, v_to):
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    
    q = np.zeros(4)
    half = np.sqrt(np.dot(v_from, v_from) * np.dot(v_to, v_to)) + np.dot(v_from, v_to)
    q[0] = half
    q[1:] = np.cross(v_from, v_to)
    q = q / np.linalg.norm(q)  # Normalize the quaternion
    
    r = R.from_quat([q[1], q[2], q[3], q[0]])  # Create a rotation object
    euler_angles = r.as_euler('xyz', degrees=True)  # Convert to Euler angles
    return euler_angles

def calculate_azimuth(arrow_direction, pelvis_position):
    camera_to_pelvis = pelvis_position[:2]  # Prendiamo solo la componente XY
    torso_direction_xy = arrow_direction[:2]  # Proiezione sul piano XY
    azimuth = np.degrees(np.arctan2(torso_direction_xy[1], torso_direction_xy[0]) - np.arctan2(camera_to_pelvis[1], camera_to_pelvis[0]))

    # Normalizza l'angolo tra 0 e 360 gradi
    azimuth = azimuth % 360
    if azimuth < 0:
        azimuth += 360

    return azimuth

def calculate_azimuth_gaze(arrow_direction, neck_position):
    camera_to_neck = neck_position[:2]  # Consideriamo solo la componente XY
    gaze_direction_xy = arrow_direction[:2]  # Proiezione sul piano XY
    azimuth_gaze = np.degrees(np.arctan2(gaze_direction_xy[1], gaze_direction_xy[0]) - np.arctan2(camera_to_neck[1], camera_to_neck[0]))

    # Normalizza l'angolo tra 0 e 360 gradi
    azimuth_gaze = azimuth_gaze % 360
    if azimuth_gaze < 0:
        azimuth_gaze += 360

    return azimuth_gaze

def should_use_kalman(azimuth):
    return (70 <= azimuth <= 110) or (250 <= azimuth <= 290)

def should_use_kalman_gaze(azimuth_gaze):
    return (70 <= azimuth_gaze <= 110) or (250 <= azimuth_gaze <= 290)

class KalmanFilter1D:
    def __init__(self, initial_state, initial_uncertainty, process_variance, measurement_variance):
        self.state = initial_state
        self.uncertainty = initial_uncertainty
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def predict(self):
        self.state = self.state  # In 1D Kalman filter, the state doesn't change during prediction
        self.uncertainty += self.process_variance

    def update(self, measurement):
        kalman_gain = self.uncertainty / (self.uncertainty + self.measurement_variance)
        self.state += kalman_gain * (measurement - self.state)
        self.uncertainty = (1 - kalman_gain) * self.uncertainty

    def get_state(self):
        return self.state

# Initialize Kalman filters
kf_position = KalmanFilter1D(
    initial_state=0.0,
    initial_uncertainty=1.0,
    process_variance=0.1,
    measurement_variance=1.0
)

kf_gaze = KalmanFilter1D(
    initial_state=0.0,
    initial_uncertainty=1.0,
    process_variance=0.1,
    measurement_variance=1.0
)

# Buffers to store the last 5 azimuth values
azimuth_buffer = []
gaze_azimuth_buffer = []

# Intrinsic parameters
intrinsics = rs.intrinsics()
intrinsics.width = 1280
intrinsics.height = 720
intrinsics.ppx = 631.7623291015625
intrinsics.ppy = 380.9401550292969
intrinsics.fx = 912.0092163085938
intrinsics.fy = 912.2039184570312
intrinsics.model = rs.distortion.inverse_brown_conrady
intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

# Initialize the YOLO detector
model = YOLO('yolov8m-pose')

# Map of body part indices to their names
index_to_label = {
    0: 'Nose', 1: 'Eye.L', 2: 'Eye.R', 3: 'Ear.L', 4: 'Ear.R',
    5: 'Shoulder.L', 6: 'Shoulder.R', 7: 'Elbow.L', 8: 'Elbow.R',
    9: 'Wrist.L', 10: 'Wrist.R', 11: 'Hip.L', 12: 'Hip.R',
    13: 'Knee.L', 14: 'Knee.R', 15: 'Ankle.L', 16: 'Ankle.R'
}

# Define keypoint connections for drawing lines
keypoint_connections = [
    (0, 1), (0, 2), (5 , 6), (11, 12), (2, 4), (1, 3), (5, 7),
    (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# Create the figure and 3D axis outside the loop
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set fixed limits for the axes
ax.set_xlim(-4, 4)
ax.set_ylim(0, 4)
ax.set_zlim(-1, 3)

# Set the axis labels to match the new orientation
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Define the vertices of a 1x1x1 cube centered at (0, 0, 0)
cube_vertices = np.array([[-0.05, -0.05, -0.05],
                          [0.05, -0.05, -0.05],
                          [0.05, 0.05, -0.05],
                          [-0.05, 0.05, -0.05],
                          [-0.05, -0.05, 0.05],
                          [0.05, -0.05, 0.05],
                          [0.05, 0.05, 0.05],
                          [-0.05, 0.05, 0.05]])

# Define the 12 edges of the cube
cube_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]

# Plot the cube
cube_lines = []
for edge in cube_edges:
    start, end = edge
    line, = ax.plot([cube_vertices[start][0], cube_vertices[end][0]],
                    [cube_vertices[start][1], cube_vertices[end][1]],
                    [cube_vertices[start][2], cube_vertices[end][2]], 'k')
    cube_lines.append(line)

# Store scatter, plot, arrow, sector, circle, and turned man text objects
scatter_plots = []
line_plots = []
text_labels = []
arrow_plots = []
sector_plots = []
circle_plots = []
april_tag_scatter = None
turned_man_text_plots = []  # To store the "Turned Man" text

# Variabili per memorizzare gli angoli azimutali
azimuth_text = None
gaze_azimuth_text = None

def extract_and_process_rosbag(bag_file):
    bridge = CvBridge()
    rgb_images = []
    depth_images = []

    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw', '/camera/depth/image_raw']):
            if topic == '/camera/color/image_raw':
                rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                rgb_images.append(rgb_image)
            elif topic == '/camera/depth/image_raw':
                depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                depth_images.append(depth_image)
    
    assert len(rgb_images) == len(depth_images), "The number of RGB and depth frames do not match."

    return rgb_images, depth_images

def detect_apriltag(color_image, depth_image):
    global april_tag_scatter

    # Define the AprilTag detector
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
    parameters = aruco.DetectorParameters()

    # Convert the image to grayscale for tag detection
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # Detect the AprilTags
    corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is not None:
        for corner, id in zip(corners, ids.flatten()):
            if id == 0:  # We're interested in tag with ID 0
                # Calculate the center of the tag in 2D
                center_x = int(np.mean(corner[0][:, 0]))
                center_y = int(np.mean(corner[0][:, 1]))

                # Convert the center point to 3D
                x_3d, y_3d, z_3d, _ = calculate_3d(center_x, center_y, depth_image, intrinsics)

                # Plot the tag center in 3D
                if april_tag_scatter:
                    april_tag_scatter.remove()
                april_tag_scatter = ax.scatter(x_3d, y_3d, z_3d, color='black', s=100)  # Black dot for the tag

                # Draw a bounding box around the detected tag on the color image
                top_left = tuple(corner[0][0].astype(int))
                bottom_right = tuple(corner[0][2].astype(int))
                cv2.rectangle(color_image, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(color_image, f"ID: {id}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

bag_file = '/home/paolo/dataset/sub5/sub5_1/mocap_data.bag'
rgb_images, depth_images = extract_and_process_rosbag(bag_file)

# Aggiungi un contatore per gli ID unici delle persone
person_id_counter = itertools.count(1)  # Inizia da 1 e incrementa per ogni persona

try:
    for color_image, depth_image in zip(rgb_images, depth_images):
        # Remove any filtering step here
        depth_filtered = depth_image

        # Run the YOLO model on the frames
        persons = model(color_image)

        # Detect AprilTag and plot in 3D
        detect_apriltag(color_image, depth_filtered)

        # Normalize depth image for display
        depth_image_display = cv2.normalize(depth_filtered, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_display = cv2.convertScaleAbs(depth_image_display)

        # Convert depth image to color for better visualization
        depth_image_colormap = cv2.applyColorMap(depth_image_display, cv2.COLORMAP_JET)

        # Remove previous scatter plots, lines, text labels, arrows, sectors, circles, and turned man text
        for scatter in scatter_plots:
            scatter.remove()
        scatter_plots.clear()
        for line in line_plots:
            line.remove()
        line_plots.clear()
        for text in text_labels:
            text.remove()
        text_labels.clear()
        for arrow in arrow_plots:
            arrow.remove()
        arrow_plots.clear()
        for sector in sector_plots:
            sector.remove()
        sector_plots.clear()
        for circle in circle_plots:
            circle.remove()
        circle_plots.clear()
        for turned_man_text in turned_man_text_plots:
            turned_man_text.remove()
        turned_man_text_plots.clear()

        all_keypoints_3d = []

        for results in persons:
            for result in results:
                if hasattr(result, 'keypoints'):
                    kpts = result.keypoints.xy.cpu().numpy()
                    keypoints_list = kpts.flatten().tolist()
                    labels = [index_to_label.get(i, '') for i in range(len(keypoints_list) // 2)]

                    keypoints_2d = {}
                    keypoints_3d = []

                    # Estrazione del bounding box per ogni persona
                    bbox = result.boxes.xyxy.cpu().numpy()[0]  # xyxy rappresenta il bounding box

                    # Assegna un ID univoco alla persona
                    person_id = next(person_id_counter)

                    # Disegna il bounding box e l'ID sul frame RGB
                    cv2.rectangle(color_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(color_image, f"Person ID: {person_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    for i, (x, y) in enumerate(zip(keypoints_list[::2], keypoints_list[1::2])):
                        cv2.circle(color_image, (int(x), int(y)), 5, (0, 255, 0), -1)
                        label = labels[i]
                        if label:
                            cv2.putText(color_image, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(color_image, f"({int(x)}, {int(y)})", (int(x) + 10, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            keypoints_2d[label] = (int(x), int(y))
                            x_3d, y_3d, z_3d, min_depth = calculate_3d(int(x), int(y), depth_filtered, intrinsics)
                            if not np.isnan(z_3d) and (x_3d != 0 or y_3d != 0 or z_3d != 0):
                                keypoints_3d.append((label, x_3d, y_3d, z_3d))

                    if 'Hip.L' in keypoints_2d and 'Hip.R' in keypoints_2d:
                        hip_l = keypoints_2d['Hip.L']
                        hip_r = keypoints_2d['Hip.R']
                        if hip_l != (0, 0) and hip_r != (0, 0):
                            pelvis_x = (hip_l[0] + hip_r[0]) // 2
                            pelvis_y = (hip_l[1] + hip_r[1]) // 2
                            keypoints_2d['Pelvis'] = (pelvis_x, pelvis_y)

                            x_3d, y_3d, z_3d, min_depth = calculate_3d(pelvis_x, pelvis_y, depth_filtered, intrinsics)
                            if not np.isnan(z_3d) and (x_3d != 0 or y_3d != 0 or z_3d != 0):
                                keypoints_3d.append(('Pelvis', x_3d, y_3d, z_3d))

                            cv2.circle(color_image, (pelvis_x, pelvis_y), 5, (0, 0, 255), -1)
                            cv2.putText(color_image, 'Pelvis', (pelvis_x, pelvis_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(color_image, f"({pelvis_x}, {pelvis_y})", (pelvis_x + 10, pelvis_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    if 'Shoulder.L' in keypoints_2d and 'Shoulder.R' in keypoints_2d:
                        shoulder_l = keypoints_2d['Shoulder.L']
                        shoulder_r = keypoints_2d['Shoulder.R']
                        if shoulder_l != (0, 0) and shoulder_r != (0, 0):
                            neck_x = (shoulder_l[0] + shoulder_r[0]) // 2
                            neck_y = (shoulder_l[1] + shoulder_r[1]) // 2
                            keypoints_2d['Neck'] = (neck_x, neck_y)

                            x_3d, y_3d, z_3d, min_depth = calculate_3d(neck_x, neck_y, depth_filtered, intrinsics)
                            if not np.isnan(z_3d) and (x_3d != 0 or y_3d != 0 or z_3d != 0):
                                keypoints_3d.append(('Neck', x_3d, y_3d, z_3d))

                            cv2.circle(color_image, (neck_x, neck_y), 5, (255, 0, 0), -1)
                            cv2.putText(color_image, 'Neck', (neck_x, neck_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(color_image, f"({neck_x}, {neck_y})", (neck_x + 10, neck_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    for (start, end) in keypoint_connections:
                        if labels[start] in keypoints_2d and labels[end] in keypoints_2d:
                            start_point = keypoints_2d[labels[start]]
                            end_point = keypoints_2d[labels[end]]
                            if start_point != (0, 0) and end_point != (0, 0):
                                cv2.line(color_image, start_point, end_point, (255, 0, 0), 2)

                    if keypoints_3d:
                        all_keypoints_3d.append({
                            'id': person_id,
                            'bbox': bbox,
                            'keypoints_3d': keypoints_3d
                        })

        # Processa ogni persona rilevata individualmente
        for person in all_keypoints_3d:
            person_id = person['id']
            bbox = person['bbox']
            keypoints_3d = person['keypoints_3d']

            for keypoint in keypoints_3d:
                label, x_3d, y_3d, z_3d = keypoint
                scatter = ax.scatter(x_3d, y_3d, z_3d)
                scatter_plots.append(scatter)

            for (start, end) in keypoint_connections:
                start_label = index_to_label.get(start, '')
                end_label = index_to_label.get(end, '')
                if start_label and end_label:
                    start_point = next((kp for kp in keypoints_3d if kp[0] == start_label), None)
                    end_point = next((kp for kp in keypoints_3d if kp[0] == end_label), None)
                    if start_point and end_point:
                        line, = ax.plot([start_point[1], end_point[1]], [start_point[2], end_point[2]], [start_point[3], end_point[3]], 'b')
                        line_plots.append(line)

            if 'Shoulder.L' in [kp[0] for kp in keypoints_3d] and 'Shoulder.R' in [kp[0] for kp in keypoints_3d] and 'Pelvis' in [kp[0] for kp in keypoints_3d] and 'Neck' in [kp[0] for kp in keypoints_3d]:
                shoulder_l = next(kp for kp in keypoints_3d if kp[0] == 'Shoulder.L')
                shoulder_r = next(kp for kp in keypoints_3d if kp[0] == 'Shoulder.R')
                pelvis = next(kp for kp in keypoints_3d if kp[0] == 'Pelvis')
                neck = next(kp for kp in keypoints_3d if kp[0] == 'Neck')

                p1 = np.array(shoulder_l[1:])
                p2 = np.array(shoulder_r[1:])
                p3 = np.array(pelvis[1:])
                p4 = p3  # La freccia parte dal punto Pelvis
                p5 = np.array(neck[1:])

                arrow_pelvis_start, arrow_pelvis_end, normal_torso = calculate_plane_and_arrow(p1, p2, p3, p4, p5, arrow_length=3)

                azimuth = calculate_azimuth(normal_torso, pelvis[1:])
                
                # Aggiungi l'azimut al buffer
                azimuth_buffer.append(azimuth)
                if len(azimuth_buffer) > 5:
                    azimuth_buffer.pop(0)

                # Verifica se l'azimut è vicino ai 90 o 270 gradi
                if should_use_kalman(azimuth):
                    for previous_azimuth in azimuth_buffer:
                        kf_position.update(previous_azimuth)

                    kf_position.predict()
                    predicted_azimuth = kf_position.get_state()

                    arrow_direction = np.array([np.cos(np.radians(predicted_azimuth)), np.sin(np.radians(predicted_azimuth)), 0])
                    arrow_end = pelvis[1:] + arrow_direction * 1.0

                    if azimuth_text:
                        azimuth_text.remove()

                    azimuth_text = ax.text2D(0.05, 0.95, f"Azimuth: {predicted_azimuth:.2f}° (Kalman)", transform=ax.transAxes, fontsize=14, color='red')
                    arrow = ax.quiver(pelvis[1], pelvis[2], pelvis[3],
                                      arrow_end[0] - pelvis[1], arrow_end[1] - pelvis[2], arrow_end[2] - pelvis[3],
                                      color='g', length=1.0, arrow_length_ratio=0.1)
                    arrow_plots.append(arrow)

                    print(f"Azimuth: {predicted_azimuth:.2f}° for Person ID: {person_id}")
                
                else:
                    arrow = ax.quiver(arrow_pelvis_start[0], arrow_pelvis_start[1], arrow_pelvis_start[2],
                                      arrow_pelvis_end[0] - arrow_pelvis_start[0], arrow_pelvis_end[1] - arrow_pelvis_start[1], arrow_pelvis_end[2] - arrow_pelvis_start[2],
                                      color='g', length=1.0, arrow_length_ratio=0.1)
                    arrow_plots.append(arrow)

            if 'Eye.L' in [kp[0] for kp in keypoints_3d] and 'Eye.R' in [kp[0] for kp in keypoints_3d] and 'Neck' in [kp[0] for kp in keypoints_3d]:
                eye_l = next(kp for kp in keypoints_3d if kp[0] == 'Eye.L')
                eye_r = next(kp for kp in keypoints_3d if kp[0] == 'Eye.R')
                neck = next(kp for kp in keypoints_3d if kp[0] == 'Neck')

                p1 = np.array(eye_l[1:])
                p2 = np.array(eye_r[1:])
                p3 = np.array(neck[1:])
                p4 = p3  # La freccia parte dal punto Neck
                p5 = np.array(neck[1:])

                arrow_neck_start, arrow_neck_end, normal_gaze = calculate_plane_and_arrow(p1, p2, p3, p4, p5, arrow_length=4)

                azimuth_gaze = calculate_azimuth_gaze(normal_gaze, neck[1:])
                gaze_azimuth_buffer.append(azimuth_gaze)
                if len(gaze_azimuth_buffer) > 5:
                    gaze_azimuth_buffer.pop(0)

                if should_use_kalman_gaze(azimuth_gaze):
                    for previous_azimuth in gaze_azimuth_buffer:
                        kf_gaze.update(previous_azimuth)

                    kf_gaze.predict()
                    predicted_gaze_azimuth = kf_gaze.get_state()

                    gaze_direction = np.array([np.cos(np.radians(predicted_gaze_azimuth)), np.sin(np.radians(predicted_gaze_azimuth)), 0])
                    arrow_end = neck[1:] + gaze_direction * 1.0

                    if gaze_azimuth_text:
                        gaze_azimuth_text.remove()
                    
                    gaze_azimuth_text = ax.text2D(0.05, 0.90, f"Gaze Azimuth: {predicted_gaze_azimuth:.2f}° (Kalman)", transform=ax.transAxes, fontsize=14, color='blue')
                    arrow = ax.quiver(neck[1], neck[2], neck[3],
                                      arrow_end[0] - neck[1], arrow_end[1] - neck[2], arrow_end[2] - neck[3],
                                      color='r', length=1.0, arrow_length_ratio=0.1)
                    arrow_plots.append(arrow)

                    print(f"Gaze Azimuth: {predicted_gaze_azimuth:.2f}° for Person ID: {person_id}")

                else:
                    arrow = ax.quiver(arrow_neck_start[0], arrow_neck_start[1], arrow_neck_start[2],
                                      arrow_neck_end[0] - arrow_neck_start[0], arrow_neck_end[1] - arrow_neck_start[1], arrow_neck_end[2] - arrow_neck_start[2],
                                      color='r', length=1.0, arrow_length_ratio=0.1)
                    arrow_plots.append(arrow)

                euler_gaze = euler_from_vectors([1, 0, 0], normal_gaze)

            # Verifica se il corpo è girato e non è possibile rilevare entrambi gli occhi
            if 'Shoulder.L' in [kp[0] for kp in keypoints_3d] and 'Shoulder.R' in [kp[0] for kp in keypoints_3d]:
                shoulder_l_x = next(kp for kp in keypoints_3d if kp[0] == 'Shoulder.L')[1]
                shoulder_r_x = next(kp for kp in keypoints_3d if kp[0] == 'Shoulder.R')[1]
                if shoulder_r_x > shoulder_l_x and ('Eye.L' not in [kp[0] for kp in keypoints_3d] or 'Eye.R' not in [kp[0] for kp in keypoints_3d]):
                    turned_man_text = ax.text2D(0.05, 0.85, f"Man Turned ID: {person_id}", transform=ax.transAxes, fontsize=8, color='blue')
                    turned_man_text_plots.append(turned_man_text)
                    print(f"Man Turned ID: {person_id}")

        plt.draw()
        plt.pause(0.001)

        cv2.imshow('YOLO Keypoints', color_image)
        cv2.imshow('Depth Image', depth_image_colormap)
        
        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"Errore: {e}")

finally:
    cv2.destroyAllWindows()

import numpy as np
import pyrealsense2 as rs


def calculate_3d(x, y, depth_frame, intrinsics, w, h, window_size=10):
    if x == 0 and y == 0:
        return 0.0, 0.0, 0.0, 0.0

    min_depth = float('inf')

    for i in range(-window_size // 2, window_size // 2 + 1):
        for j in range(-window_size // 2, window_size // 2 + 1):
            x_pixel = int(x) + i
            y_pixel = int(y) + j

            if 0 <= x_pixel < w and 0 <= y_pixel < h:
                depth = depth_frame.get_distance(x_pixel, y_pixel)
                if depth != 0 and depth < min_depth:
                    min_depth = depth

    if min_depth == float('inf'):
        return 0.0, 0.0, 0.0, 0.0

    pixel = [x, y]
    depth = min_depth

    # Convert 2D pixel coordinates to 3D point in camera space
    point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth)

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

    p5_xy = np.array([p5[0], p5[1], 0])
    angleX_camera_body = np.arctan2(p5[1], p5[0])
    angleX_camera_body_degrees = np.degrees(angleX_camera_body)

    def line_intersection(p1, p2, p3, p4):
        a1, b1 = p1[:2]
        a2, b2 = p2[:2]
        a3, b3 = p3[:2]
        a4, b4 = p4[:2]

        denominator = (a1 - a2) * (b3 - b4) - (b1 - b2) * (a3 - a4)
        if denominator == 0:
            return None

        intersection_x = ((a1 * b2 - b1 * a2) * (a3 - a4) - (a1 - a2) * (a3 * b4 - b3 * a4)) / denominator
        intersection_y = ((a1 * b2 - b1 * a2) * (b3 - b4) - (b1 - b2) * (a3 * b4 - b3 * a4)) / denominator
        return np.array([intersection_x, intersection_y, 0])

    intersection = line_intersection([0, 0, 0], p5, arrow_start, arrow_end)

    if intersection is not None:
        orientation = np.arctan2(intersection[1], intersection[0])
        orientation_degrees = np.degrees(orientation)
    else:
        orientation_degrees = None

    return arrow_start, arrow_end, angleX_camera_body_degrees, orientation_degrees


def calculate_azimuth(arrow_direction, pelvis_position):
    camera_to_pelvis = pelvis_position[:2]  # We only take the XY component
    torso_direction_xy = arrow_direction[:2]  # Projection onto the XY plane
    azimuth = np.degrees(np.arctan2(torso_direction_xy[1], torso_direction_xy[0]) - np.arctan2(camera_to_pelvis[1], camera_to_pelvis[0]))

    # Normalize the angle between 0 and 360 degrees
    azimuth = azimuth % 360
    if azimuth < 0:
        azimuth += 360

    return azimuth


def calculate_azimuth_gaze(arrow_direction, neck_position) -> float:
    camera_to_neck = neck_position[:2]  # We only consider the XY component
    gaze_direction_xy = arrow_direction[:2]  # Projection onto the XY plane
    azimuth_gaze = np.degrees(np.arctan2(gaze_direction_xy[1], gaze_direction_xy[0]) - np.arctan2(camera_to_neck[1], camera_to_neck[0]))

    # Normalize the angle between 0 and 360 degrees
    azimuth_gaze = azimuth_gaze % 360
    if azimuth_gaze < 0:
        azimuth_gaze += 360

    return azimuth_gaze


def should_use_kalman(azimuth):
    # Determines if Kalman filter should be used based on azimuth angle
    return (70 <= azimuth <= 110) or (250 <= azimuth <= 290)


def should_use_kalman_gaze(azimuth_gaze):
    # Determines if Kalman filter should be used for gaze based on azimuth angle
    return (70 <= azimuth_gaze <= 110) or (250 <= azimuth_gaze <= 290)

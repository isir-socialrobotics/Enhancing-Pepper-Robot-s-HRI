# Enhancing-Pepper-Robot-s-HRI
Enhancing Pepper Robot’s Human-Robot Interaction Capabilities through Advanced Hardware Integration and Human Pose Estimation
This repository contains two folders:

1. **3D Printing Files**:
    - Custom-designed box to house both the GPU and the battery
    <div align="center">
        <img src="https://github.com/polmagri/Enhancing-Pepper-Robot-s-HRI/assets/150929375/7c31d37e-d354-4a25-9d5d-cb75ebc4d5ee" alt="cad_box (1)">
    </div>
    - RealSense D435i Camera Mount
    <div align="center">
        <img src="https://github.com/polmagri/Enhancing-Pepper-Robot-s-HRI/assets/150929375/83262071-3eee-4a77-b298-98f52fd20e6f" alt="cad_camera (1)">
    </div>


2. **Dataset**:
   
   - [Link drive to download Dataset](https://drive.google.com/drive/folders/1_3JckYWL6bLGh8cu_JtG2LzMEAdoCGat?usp=sharing)
   
    Composed of 5 different subjects performing a total of 8 trials using a motion capture system with 8 cameras in a closed environment with standard lighting. The dataset simulates various types of human movements, capturing different distances, orientations, heights, and poses. Each trial lasts approximately 2 minutes with a single person in the room.
    The various scenes were also recorded simultaneously with a RealSense camera, positioned statically at a height of 1.25 meters to obtain the same view as when Pepper is in an upright position.

The analyzed scenes include:
- Standard walk back and forth
- Walk back and forth with arms crossed
- Walk back and forth with sudden movements (e.g., dodging an obstacle)
- Walk back and forth in a zigzag pattern with pronounced head movements

![dataset_acquisition (1)](https://github.com/polmagri/Enhancing-Pepper-Robot-s-HRI/assets/150929375/c2b1b9d7-705d-4a66-8adf-e0cef52e414a)

3. The **FOV_dataset** code written in Python is used to test the algorithm on the dataset. It returns a green arrow to indicate torso orientation, a red arrow with a 120-degree arc for gaze orientation, and a warning message "turned man + ID bounding box" to signal when the robot is outside the human's field of view.
Due to the loss of quality in the depth images during the dataset saving process, techniques have been implemented to distinguish the background from the foreground:

A double-window filter around the keypoints to select the minimum depth value and a threshold for the exclusion of depth values greater than 4 meters.


4. The **FOV_realsense** Python code is the version of the code to be used directly with the connected Realsense camera. It utilizes the Realsense functions to obtain intrinsic parameters, applies a single-window technique to improve depth quality, and has no limitations on depth. Like the previous algorithm, it uses a Kalman filter for both gaze and body position in more critical situations (e.g., when the person is in profile relative to the camera, specifically at azimuth angles of 90 and 270 degrees ± 20 degrees) to correct errors due to the misalignment of the keypoints for the back and gaze

## Documentation 

You can find the detailed documentation in the following file:

- [PDF Document](https://github.com/polmagri/Enhancing-Pepper-Robot-s-HRI/blob/main/Enhancing_Pepper_Robot_s_HRI.pdf)

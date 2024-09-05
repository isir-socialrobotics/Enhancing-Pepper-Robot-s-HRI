# Enhancing-Pepper-Robot-s-HRI
Enhancing Pepper Robot’s Human-Robot Interaction Capabilities through Advanced Hardware Integration and Human Pose Estimation
This repository contains two folders:

1. **3D Printing Files**:

    - Custom-designed box to house both the GPU and the battery
    - RealSense D435i Camera Mount
![finale (2)](https://github.com/user-attachments/assets/30f2dce7-f90b-40ff-9658-3d65164607c4)



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

3. The `FOV_dataset.py` code, in src folder, written in Python is used to test the algorithm on the dataset. It returns a green arrow to indicate torso orientation, a red arrow with a 120-degree arc for gaze orientation, and a warning message "turned man + ID bounding box" to signal when the robot is outside the human's field of view.
Due to the loss of quality in the depth images during the dataset saving process, techniques have been implemented to distinguish the background from the foreground:

A double-window filter around the keypoints to select the minimum depth value and a threshold for the exclusion of depth values greater than 4 meters.

![finalfoto (2)](https://github.com/user-attachments/assets/91cc006b-13fa-4115-97f5-8ecf9da270c0)



4. The fov_online folder contains the version of the code implemented in ROS Noetic, ready to be used directly with a connected RealSense camera. It leverages RealSense functions to obtain intrinsic parameters and applies a single-window technique to enhance depth quality without any depth limitations. Like the previous algorithm, it uses a Kalman filter for both gaze and body position in more critical situations (e.g., when the person is in profile relative to the camera, specifically at azimuth angles of 90 and 270 degrees ± 20 degrees) to correct errors caused by the misalignment of keypoints for the back and gaze.
As a publisher, it includes:

- 3D coordinates of human keypoints
- Warning messages if the person is turned away and cannot be detected
- Azimuth angles (relative to the camera) for gaze direction and body orientation

![image](https://github.com/user-attachments/assets/ca71a4b0-4b1e-4b79-b0f8-a900810652a5)

5. 
## Documentation 

You can find the detailed documentation in the following file:

- [PDF Document](https://arxiv.org/pdf/2409.01036)

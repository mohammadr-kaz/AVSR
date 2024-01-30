# AVSR
Autonomous Vision-based Space Rendezvous
This is a simulation of vision-based space rendezvous in proximity operation between a chaser cubesat and a target cubsat. There is a LiDAR on chaser cubesat which provides point-cloud data of the target.


An improved RANSAC method is developed in order to perform pose estimation of the target. This method not only is faster than RANSAC but also in terms of precision performs well.

## Prerequisites

- 1. ROS Noetic
- 2. pyransac3d
- 3. OpenCV
- 4. numpy
- 5. CvBridge

## Instruction to Run

1. create a ros workspace
2. create "src", "devel", and "build" directories inside the workspace
3. clone the repository inside "src" directory
4. use  "catkin_make" in order to make the package
4. open terminal inside the workspace and enter this code:
```
source devel/setup.bash
```
5. enter the following line in the terminal
```
roslaunch docking simulation.launch
```
6. in another terminal enter the following lines:
```
source devel/setup.bash
rosrun docking pcl_to_array.py
```
7. at last you can run matlab simulink model and run it.
(the matlab model is not uploaded (maybe in future as it is a modular package))
The performance of the rendezvous can be seen in gazebo.


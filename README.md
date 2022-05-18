### Capstone-Prototype
 
This repository contains a portion of our team's code for the ROB498 Capstone project. Our project is a prototype of a computer-vision object tracking system for the OAK-D stereo camera.

The stereo camera's video is input to ORB-SLAM v3 to localize the camera's position, as well as to YOLO v4 for object recognition.

Detected objects are transformed from the camera frame of reference to an inertial frame of reference. The inertial frame pose of each detected object is continuously updated using input and output buffers as well as an EMA filter.

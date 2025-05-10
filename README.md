# -Tennis-Ball-Wall-Impact-Detection-System

Overview

This project applies a light computer vision system that:

Identifies a tennis ball impact on a vertical wall via a live webcam.

Maps and traces the impact location from the real world to a virtual screen through homography.

Runs in real-time on low-spec hardware, without dependence on deep learning models.

The system is made with simplicity, efficiency, and resourcefulness in mind — and hence is suitable for edge devices and live coaching environments.


Goals:
Implement real-time tracking without the use of GPU acceleration.

Identify time and place of impact on a horizontal wall surface.

Show virtual impact points for visualization or examination.

Core Concepts:

Employ OpenCV's MOG2 background subtraction to identify high-speed moving objects.

Blobs filter with contour area and radius to estimate the size of the ball.

Homography transform to map actual hits onto a virtual canvas.

Implement a cooldown feature to prevent repeated detections for one hit.


Key Elements:

Element	Description

BackgroundSubtractorMOG2	Finds movement in the frame
cv2.findContours()	Locates candidate hit points
cv2.minEnclosingCircle()	Filters the object size and shape
cv2.findHomography() + cv2.perspectiveTransform()	Maps the hit to a virtual coordinate space
virtual_screen	Window where all hits are marked in real time
CSV Logger	Saves time-stamped hit data for analysis


Live Webcam then  Frame Processing then Motion Detection then Hit Detection then Homography Mapping then Virtual Screen and Logging


Challenges Overcome

1. Prevention of False Positives
Basic motion detection caught shadows and body movement.

Solved through morphological filtering and contour size thresholds.

2. Accurate Mapping of Hits
Required accurate mapping from world coordinates to a view on the screen.

Solved through a 4-point perspective transform using cv2.findHomography().

3. Hit Debouncing
The ball would register multiple hits if it was stationary near the wall.

Solved through a frame-based cooldown (e.g. 15-frame delay between hits).

4. Hardware Limitations
The system needed to function well without GPU or neural networks.

Didn't use deep learning at all in favor of geometry + motion.

Learnings

How to achieve real-time motion tracking with limited resources.

Practical application of homography matrices to register two planes (wall ↔ screen).

Effective means of visualizing and logging impact data in live environments.

The significance of modularity, threshold adjustment, and testing in CV systems.

Learned to balance accuracy against performance using intelligent filtering.




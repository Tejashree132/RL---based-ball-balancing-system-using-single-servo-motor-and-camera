# RL---based-ball-balancing-system-using-single-servo-motor-and-camera
This project uses Reinforcement Learning to balance a ball on a beam by adjusting the beam’s angle via a servo motor. A camera tracks the ball’s position, giving real-time feedback to the RL agent, which learns through rewards to maintain stability. It combines control, vision, and machine learning.

Abstract:
This project aims to  build a system to balance a ball on a platform using a single servo motor and a camera.
A Reinforcement Learning (RL) algorithm learns to move the platform to keep the ball centered.
The camera tracks the ball's position, and the system adjusts the platform  based on what it learns over time.

Components Required
Hardware Requirements
Servo Motor, Camera, Ball ,Mounting Rod, Pipe,base plate-By 3D Printing machine,Raspberry Pi
Software Requirements
Model free RL algorithm.For Design - Auto CAD, Cura Software,3D printing software,Gymnasium Environment

Applications
Robotics and Automation: This project improves robotic systems for precise control and adaptability, helping them maintain balance on uneven terrain or handle delicate tasks.
Example -   walking robot that adjusts its leg movements to balance on rough or shifting surfaces.

Stabilization Systems : The techniques in this project can  help to develop balance control in drones, allowing them to adapt and stay stable for safer, more efficient operation.
Example -  drone equipped with a camera that uses RL to stabilize its flight during windy conditions by adjusting its rotor speeds

Model Free RL System :
RL agent does not try to understand the environment dynamics. Instead, it builds a guide (policy) for itself that tells what the optimal behavior in a is given state i.e. the best action to be taken in a given state. This is built using error and trial methods by the agent.
The focus is on learning by observing the consequences of actions rather than attempting to understand the dynamics of the environment.
Consider the game of cards - We have a handful of cards in our hand, and we must pick one card to play. Here instead of thinking of all possible future outcomes associated with playing each card which is nearly impossible to model the agent will try to learn what is the best card to play given the current hands of the card based on its interaction with the environment.

Limitation and Future Scope:
Camera Frame Rate latency is high: Video capture rate slows the results  delayed action.
Raspberry Pi architecture has limited power to process computer vision applications, Nvidia Jetson can improve further. 
Servo Quality could be further  with improv PID control.
Deep Q-Learning can be for better results 
Mass of the ball and coefficient of friction are not being taken into consideration of the code for balancing the ball 

Delay Quantitative Analysis:
Average time per frame: 0.6078 seconds
Estimated servo delay: 0.4000 seconds per move
Total processing time: 66.85 seconds  
Total frames captured: 110

Camera capturing is slow and has delay/lags compared to real ball movement.
Your camera is running at 720p (1280x720 resolution), high
resolution means heavier frames, which causes processing delay.
Raspberry Pi 4B, even though it's good (Quad-core Cortex-A72 @ 1.5GHz), can't quickly handle 720p + OpenCV + servo movements at the sametime without some optimization.
So the real reasons of lag are:
720p frame is big (high pixel data to process).
Using HoughCircles is computationally heavy.
OpenCV on Raspberry Pi works much faster on lower resolution like 480p or 320p.
Also cv2.imshow() adds some latency when showing frames.
Total running time: 27.49 seconds
Average servo response delay: 0.4237 seconds
Ball disappeared at 07:57:38, Duration: 5.31s
Ball appeared at 07:57:38
Ball disappeared at 07:57:40, Duration: 1.84s
Ball appeared at 07:57:40
Ball disappeared at 07:57:41, Duration: 0.97s
Ball appeared at 07:57:41
The ball is disappearing from the frame as the cv2 window is not able to sense it via the camera frame high rate response thus after reappearing the servo is experiencing delay in the time to act and there is continuous see -saw action happening at servo arm

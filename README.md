## Posefall
This repository contains two fall detection projects. 

The first one is a video based fall detection approach https://arxiv.org/abs/2107.14633, which takes 300 frames of 2D poses to detect whether there is a fall behaviour. The experiments were conducted on NTU RGBD dataset, we refer to codes from https://github.com/kchengiva/DecoupleGCN-DropGraph to train and evaluate our model. Only codes for our fall detection model is provided in this repository because...I'm lazy... Cleaning up the codes is quite time-comsuming and my research interest is not action recognition.

The other is a fall detection system on ROS based on single frame 2D pose. Here is the demo: https://www.bilibili.com/video/BV1AT4y1u7t3/ 
Pretrained weights can be found here: https://drive.google.com/drive/folders/1s1hcetDzHP6DlVecV3Zj199TVsDVsjLI?usp=sharing

I do not plan to update this repository detaily, you can send messages to me if you are interested in this work.

Tips for "ROS_fall_detection"
Remember to modify the first line in "cam.py" and "detector.py": #! /home/seanchen/anaconda3/bin/python.
Change it to your own Python3 path as our rep uses pytorch with Python3, while original ros is based on Python2.

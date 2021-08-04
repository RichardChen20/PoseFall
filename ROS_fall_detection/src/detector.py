#! /home/seanchen/anaconda3/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#import sys
import rospy
from std_msgs.msg import String

import torch
import torch.nn.parallel
import torch.nn.functional as F
import numpy as np
import cv2
from LPN import LPN
from fall_net import Fall_Net
from pose_utils import Cropmyimage
from pose_utils import Drawkeypoints
import plot_sen
from time import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
global cam_image
def callback(data):
    try:
        global cam_image
        cam_image = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, -1))
        #print(cam_image.shape)
        # show_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)


if __name__ == '__main__':
    rospy.init_node('detector', anonymous=True)
    pub = rospy.Publisher('det_result', Image, queue_size=10)
    rospy.Subscriber('cam_image', Image, callback)
    rate = rospy.Rate(50) # 10hz
    # model 
    pose_net = LPN(nJoints=17)
    pose_net.load_state_dict(torch.load('/home/seanchen/robot_fall_det/pose_net_pred100.pth.tar'))
    pose_net.cuda()
    fall_net = Fall_Net(64, 48, 17, device=torch.device('cuda'))
    fall_net.cuda().double()
    fall_net.load_state_dict(torch.load('/home/seanchen/robot_fall_det/fall_net_pred5.pth.tar'))
    pose_net.eval()
    fall_net.eval()
    print('Load successfully!')

    bridge = CvBridge()
    global cam_image
    cam_image = np.array([])
    fall_count = []
    while not rospy.is_shutdown():
        rate.sleep()
        if not cam_image.any():
            print('waiting!')
            continue
        start = time()
        # 每来一张图检测一次，更新显示
        # image initialize
        #photo_file = '/home/seanchen/robot_fall_det/fall1.jpg'
        #input = cv2.imread(photo_file)# cv2 返回np.array类型，为(w,h,channel)
        input = cam_image
        bbox = [0, 0, input.shape[1], input.shape[0]]
        input_image, details = Cropmyimage(input, bbox)
        input_image = np.array([input_image.numpy()])
        #print(input_image.shape)
        input_image = torch.from_numpy(input_image)
        #input_image.cuda()
        # get posedetails
        pose_out = pose_net(input_image.cuda())
        fall_out, pose_cor = fall_net(pose_out)
        # 跌倒结果计算
        # 姿态可视化
        neck = (pose_cor[:, 5:6, :] + pose_cor[:, 6:7, :]) / 2
        pose_cor = torch.cat((pose_cor, neck), dim=1)
        pose_cor = pose_cor * 4 + 2.
        scale = torch.Tensor([[256, 192]]).cuda()
        pose_cor = pose_cor / scale
        scale = torch.Tensor([[details[3]-details[1], details[2]-details[0]]]).cuda()
        pose_cor = pose_cor * scale
        scale = torch.Tensor([[details[1], details[0]]]).cuda()
        pose_cor = pose_cor + scale
        #pose_cor_1 = (4*pose_cor[:, :, 0]+2.)/64*(details[3]-details[1])/4+details[1]
        #pose_cor_2 = (4*pose_cor[:, :, 1]+2.)/48*(details[2]-details[0])/4+details[0]
        pose_cor = torch.flip(pose_cor, dims=[2])
        ones = torch.ones(1, 18, 1).cuda()
        pose_cor = torch.cat((pose_cor, ones), dim=2).cpu().detach().numpy()
        #det_result = torch.zeros(64, 48, 3).numpy()
        det_result = plot_sen.plot_poses(input, pose_cor)
        #print(det_result.shape)
        # 跌倒估计
        #if fall_out.indices == 1:
        #    print('Down!')
        #if fall_out.indices == 0:
        #    print('Not Down!')
        fall_out = torch.max(F.softmax(fall_out, dim=0), dim=0)
        fall_count.append(fall_out.indices)
        fall_dis = sum(fall_count[len(fall_count)-30 : len(fall_count)])
        #print(len(fall_count))
        end = time()
        run_time = end-start
        if fall_dis > 24:
            print('Normal!', 1. / run_time)
        else:
            print('Down!', 1. / run_time)
        det_result = bridge.cv2_to_imgmsg(det_result, encoding="passthrough")
        pub.publish(det_result)
    
        
        #print(1. / run_time)
    
    # spin() simply keeps python from exiting until this node is stopped
    #rospy.spin()
    #while True:
        #pass

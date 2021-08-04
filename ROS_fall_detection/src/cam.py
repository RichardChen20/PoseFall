#! /home/seanchen/anaconda3/bin/python
import rospy
from std_msgs.msg import String
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from time import *

global show_image

def det_callback(data):
    try:
        global show_image
        show_image = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, -1))
        #print(show_image.shape)
        # show_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    #cv2.imshow("Result window", show_image)


def image_pub():
    pub = rospy.Publisher('cam_image', Image, queue_size=10)
    sub = rospy.Subscriber('det_result', Image, det_callback)
    rospy.init_node('cam', anonymous=True)
    rate = rospy.Rate(100) # 10hz
    global show_image
    show_image = np.array([])

    cameraCapture = cv2.VideoCapture(-1)
    #cv2.namedWindow('Camera')
    bridge = CvBridge()

    while not rospy.is_shutdown():
        time_str = "time %s" % rospy.get_time()
        #rospy.loginfo(time_str)
        success, frame = cameraCapture.read()
        #cv2.imshow('Camera', frame)
        while success and cv2.waitKey(1) == -1:
            start = time()
            #cv2.imshow('Camera', frame)
            success, frame = cameraCapture.read()
            image_message = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
            pub.publish(image_message)
            #rate.sleep()
            #global show_image
            if show_image.any():
                cv2.imshow('Camera', show_image)
                #cv2.imshow('Capture', frame)
            end = time()
            run_time = end-start
            print(1. / run_time)
    cv2.destroyAllWindows('Camera')
    cameraCapture.release()
        
        

if __name__ == '__main__':
    try:
        image_pub()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows('Camera')

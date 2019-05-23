#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Nils

"""Extract images from a rosbag.
"""


import cv2
#import cv2.cv2 as cv
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag


### Global Variables
g_bridge = CvBridge()
g_img = None
g_draw = None
g_img_count = 0

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    print "Extract images from %s on topic %s into %s" % (args.bag_file,
                                                          args.image_topic, args.output_dir)

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        cv2.imwrite(os.path.join(args.output_dir, "frame%06i.png" % count), cv_img)
        print "Wrote image %i" % count

        count += 1

    bag.close()

    return

def main2():
    # ROSbag to load:
    # rosbag_images_sorted.bag
    # Rosbag_CandyShop2_neu.bag
    rosbag_file = "Rosbag_CandyShop2_neu3.bag"  #Rosbag_CandyShop2_neu.bag
    bag = rosbag.Bag(rosbag_file, "r")
    bridge = CvBridge()

    #types:       sensor_msgs/Image [060021388200f6f0f447d0fcd9c64743]
    #topics:      /camera/image_raw   75 msgs    : sensor_msgs/Image

    topic_name = '/camera/image_raw'
    current_ir_img = None

    try:
        cvbridge = CvBridge()
        for topic, msg, t in bag.read_messages(topic_name):
            #print msg
            #current_ir_img = cvbridge.imgmsg_to_cv2(msg)
            #cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            global g_img_count
            g_img = g_bridge.imgmsg_to_cv2(msg, "rgb8")
            #name = "/home/nils/catkin_ws/src/dope/tmp/img" + str(g_img_count) + ".png"
            name = "/media/nils/Ubuntu-TMP/Bag_To_Images/img" + str(g_img_count) + ".png"

            g_img_count += 1
            cv2.imwrite(name, cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB))  # for debugging
            # print("Image:" + str(g_img_count))


    finally:
        bag.close()

if __name__ == '__main__':
    main2()

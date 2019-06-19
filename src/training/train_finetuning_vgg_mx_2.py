# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
# Edited by Ludeke

from __future__ import print_function

######################################################
"""
REQUIREMENTS:
matplotlib==2.2.2
simplejson==3.16.0
numpy==1.14.1
opencv_python==3.4.3.18
horovod==0.13.5
photutils==0.5
scipy==1.1.0
torch==0.4.0
pyquaternion==0.9.2
tqdm==4.25.0
pyrr==0.9.2
Pillow==5.2.0
torchvision==0.2.1
PyYAML==3.13
"""

######################################################
"""
HOW TO TRAIN DOPE

This is the DOPE training code.  
It is provided as a convenience for researchers, but it is otherwise unsupported.

Please refer to `python train.py --help` for specific details about the 
training code. 

If you download the FAT dataset 
(https://research.nvidia.com/publication/2018-06_Falling-Things)
you can train a YCB object DOPE detector as follows: 

```
python train.py --data path/to/FAT --object soup --outf soup 
--gpuids 0 1 2 3 4 5 6 7 
```

This will create a folder called `train_soup` where the weights will be saved 
after each epoch. It will use the 8 gpus using pytorch data parallel. 
"""


import argparse
import ConfigParser
import os
import random
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.models as models
import datetime
import json
import glob
import os
import copy

from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageEnhance

from math import acos
from math import sqrt
from math import pi

from os.path import exists, basename
from os.path import join

import cv2
import colorsys,math

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

# Image Show------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# Print Table Loss -----------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import csv

# Import DOPE code for Metric ------------------------------------------------------------------------------------------
from cv_bridge import CvBridge, CvBridgeError
import yaml
import rospy
import rospkg
import sys
rospack = rospkg.RosPack()
g_path2package = rospack.get_path('dope')
sys.path.append("{}/src/inference".format(g_path2package))
from cuboid import *
from detector import *

#ros msg
#from geometry_msgs.msg import PoseStamped

### Global Variables
g_bridge = CvBridge()

##################################################
# NEURAL NETWORK MODEL
##################################################

class DopeNetwork(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
    ):
        super(DopeNetwork, self).__init__()

        self.stop_at_stage = stop_at_stage

        if pretrained is False:
            print("Training network without imagenet weights.")
        else:
            print("Training network pretrained on imagenet.")

        vgg_full = models.vgg19(pretrained=pretrained).features
        self.vgg = nn.Sequential()
        for i_layer in range(24):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        # Add some layers Conv2d wieso = https://cdn-images-1.medium.com/max/1600/1*aBN2Ir7y2E-t2AbekOtEIw.png
        i_layer = 23
        self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer + 1), nn.ReLU(inplace=True))
        self.vgg.add_module(str(i_layer + 2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer + 3), nn.ReLU(inplace=True))

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        self.m1_2 = DopeNetwork.create_stage(128, numBeliefMap, True)
        self.m2_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m3_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m4_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m5_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m6_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)

        # print('---Affinity----------------------------------------------')
        # _1 are the affinity map stages
        self.m1_1 = DopeNetwork.create_stage(128, numAffinity, True)
        self.m2_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m3_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m4_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m5_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m6_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)

    def forward(self, x):
        '''Runs inference on the neural network'''

        out1 = self.vgg(x)

        out1_2 = self.m1_2(out1)
        out1_1 = self.m1_1(out1)

        if self.stop_at_stage == 1:
            return [out1_2], \
                   [out1_1]

        out2 = torch.cat([out1_2, out1_1, out1], 1)
        out2_2 = self.m2_2(out2)
        out2_1 = self.m2_1(out2)

        if self.stop_at_stage == 2:
            return [out1_2, out2_2], \
                   [out1_1, out2_1]

        out3 = torch.cat([out2_2, out2_1, out1], 1)
        out3_2 = self.m3_2(out3)
        out3_1 = self.m3_1(out3)

        if self.stop_at_stage == 3:
            return [out1_2, out2_2, out3_2], \
                   [out1_1, out2_1, out3_1]

        out4 = torch.cat([out3_2, out3_1, out1], 1)
        out4_2 = self.m4_2(out4)
        out4_1 = self.m4_1(out4)

        if self.stop_at_stage == 4:
            return [out1_2, out2_2, out3_2, out4_2], \
                   [out1_1, out2_1, out3_1, out4_1]

        out5 = torch.cat([out4_2, out4_1, out1], 1)
        out5_2 = self.m5_2(out5)
        out5_1 = self.m5_1(out5)

        if self.stop_at_stage == 5:
            return [out1_2, out2_2, out3_2, out4_2, out5_2], \
                   [out1_1, out2_1, out3_1, out4_1, out5_1]

        out6 = torch.cat([out5_2, out5_1, out1], 1)
        out6_2 = self.m6_2(out6)
        out6_1 = self.m6_1(out6)

        return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2], \
               [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]

    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        '''Create the neural network layers for a single stage.'''

        model = nn.Sequential()
        mid_channels = 128
        if first:
            padding = 1
            kernel = 3
            count = 6
            final_channels = 512
        else:
            padding = 3
            kernel = 7
            count = 10
            final_channels = mid_channels

        # First convolution
        model.add_module("0",
                         nn.Conv2d(
                             in_channels,
                             mid_channels,
                             kernel_size=kernel,
                             stride=1,
                             padding=padding)
                         )

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            model.add_module(str(i),
                             nn.Conv2d(
                                 mid_channels,
                                 mid_channels,
                                 kernel_size=kernel,
                                 stride=1,
                                 padding=padding))
            i += 1

        # Penultimate convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        i += 1

        # Last convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
        i += 1

        return model


##################################################
# UTILS CODE FOR LOADING THE DATA
##################################################

def default_loader(path):
    return Image.open(path).convert('RGB')


# Rotationsmatrix aus Quaternion
# Die Rotationsmatrix kann aus der Quaternion berechnet werden, falls sie benoetigt wird.
def QtoR(q):
    '''Calculates the Rotation Matrix from Quaternion
    a is the real part
    b, c, d are the complex elements'''
    # Source: Buchholz, J. J. (2013). Vorlesungsmanuskript Regelungstechnik und Flugregler.
    # GRIN Verlag. Retrieved from http://www.grin.com/de/e-book/82818/regelungstechnik-und-flugregler
    # https://www.cbcity.de/tutorial-rotationsmatrix-und-quaternion-einfach-erklaert-in-din70000-zyx-konvention
    q = normQ(q)

    a, b, c, d = q

    R11 = (a ** 2 + b ** 2 - c ** 2 - d ** 2)
    R12 = 2.0 * (b * c - a * d)
    R13 = 2.0 * (b * d + a * c)

    R21 = 2.0 * (b * c + a * d)
    R22 = a ** 2 - b ** 2 + c ** 2 - d ** 2
    R23 = 2.0 * (c * d - a * b)

    R31 = 2.0 * (b * d - a * c)
    R32 = 2.0 * (c * d + a * b)
    R33 = a ** 2 - b ** 2 - c ** 2 + d ** 2

    # https://www.matheplanet.com/matheplanet/nuke/html/viewtopic.php?topic=178732
    # https://github.com/balzer82/RotationMatrix/blob/master/RotationMatrix-und-Quaternion.py
    #return np.matrix([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]]) -> How it should be, but it is:
    return np.matrix([[R13, R12, R11*-1], [R33, R32, R31*-1], [R23, R22, R21*-1]])

def normQ(q):
    '''Calculates the normalized Quaternion
    a is the real part
    b, c, d are the complex elements'''
    # Source: Buchholz, J. J. (2013). Vorlesungsmanuskript Regelungstechnik und Flugregler.
    # GRIN Verlag. Retrieved from http://www.grin.com/de/e-book/82818/regelungstechnik-und-flugregler
    a, b, c, d = q

    # Betrag
    Z = np.sqrt(a ** 2 + b ** 2 + c ** 2 + d ** 2)

    return np.array([a / Z, b / Z, c / Z, d / Z])


def loadjson(path, objectsofinterest, img):
    """
    Loads the data from a json file.
    If there are no objects of interest, then load all the objects.
    """
    with open(path) as data_file:
        data = json.load(data_file)
    #print (path)
    #print("loadjson:data: ", data)

    pointsBelief = []
    boxes = []
    points_keypoints_3d = []
    points_keypoints_2d = []
    pointsBoxes = []
    poses = []
    centroids = []

    translations = []
    rotations = []
    points = []

    pose_transform = []

    #projected_cuboid = []

    #print("loadjson:data['objects'] " + data['objects'])
    for i_line in range(len(data['objects'])):
        info = data['objects'][i_line]
        if not objectsofinterest is None and \
                not objectsofinterest.lower() in info['class'].lower(): #Both lower or this is not working
            continue

        #print("loadjson:class: "+ info['class'].lower() + " objectsofinterest: " + objectsofinterest.lower())
        box = info['bounding_box']
        boxToAdd = []

        boxToAdd.append(float(box['top_left'][0]))
        boxToAdd.append(float(box['top_left'][1]))
        boxToAdd.append(float(box["bottom_right"][0]))
        boxToAdd.append(float(box['bottom_right'][1]))
        boxes.append(boxToAdd)

        boxpoint = [(boxToAdd[0], boxToAdd[1]), (boxToAdd[0], boxToAdd[3]),
                    (boxToAdd[2], boxToAdd[1]), (boxToAdd[2], boxToAdd[3])]

        pointsBoxes.append(boxpoint)


        # 3dbbox with belief maps
        points3d = []

        pointdata = info['projected_cuboid']
        for p in pointdata:
            points3d.append((p[0], p[1]))

        # Get the centroids
        pcenter = info['projected_cuboid_centroid']

        points3d.append((pcenter[0], pcenter[1]))
        pointsBelief.append(points3d)
        points.append(points3d + [(pcenter[0], pcenter[1])])
        centroids.append((pcenter[0], pcenter[1]))

        #save 3d keypoints? test:
        points_keypoints_3d.append(points3d)
        #print("loadjson:points_keypoints_3d: " , points_keypoints_3d)

        # load translations
        location = info['location']
        translations.append([location[0], location[1], location[2]])

        # quaternion
        rot = info["quaternion_xyzw"]
        rotations.append(rot)

        # todo: Change for old and new NDDS
        # load pose_transform
        pose_transform = info["pose_transform"] #NEW NDDS
        # pose_transform = info["pose_transform_permuted"] #OLD NDDS FOR FAT DATASET
        #print("pose_transform: ", pose_transform)
        pose_transform = np.asarray(pose_transform)
        #print("numpy_pose_transform: ", numpy_pose_transform)

    return {
        "pointsBelief": pointsBelief,
        "rotations": rotations,
        "translations": translations,
        "pose_transform": pose_transform,
        "centroids": centroids,
        "points": points,
        "keypoints_2d": points_keypoints_2d,
        "keypoints_3d": points_keypoints_3d,
    }


def loadimages(root):
    """
    Find all the images in the path and folders, return them in imgs.
    """
    imgs = []

    def add_json_files(path, ):
        for imgpath in glob.glob(path + "/*.jpg"):
            #print("imgpath: ", imgpath)
            #print("imgpath:exists: ", exists(imgpath))
            #test = exists(imgpath.replace('jpg', "json"))
            #print("imgpath:exists:replace: ", test)
            if exists(imgpath) and exists(imgpath.replace('jpg', "json")):  # wenn zum zugehoerigen png Bild noch ein JSON existiert dann lade es?
                imgs.append((imgpath, imgpath.replace(path, "").replace("/", ""),
                             imgpath.replace('jpg', "json")))

        for imgpath in glob.glob(path + "/*.png"):
            #print("imgpath: ", imgpath)
            #print("imgpath:exists: ", exists(imgpath))
            #test = exists(imgpath.replace('png', "json"))
            #print("imgpath:exists:replace: ", test)
            if exists(imgpath) and exists(imgpath.replace('png', "json")):  # wenn zum zugehoerigen png Bild noch ein JSON existiert dann lade es?
                imgs.append((imgpath, imgpath.replace(path, "").replace("/", ""),
                             imgpath.replace('png', "json")))

    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [os.path.join(path, o) for o in os.listdir(path)
                   if os.path.isdir(os.path.join(path, o))]
        #print("folders: ", len(folders))
        if len(folders) > 0: # recursiv durch die darunterliegenden Ordner
            for path_entry in folders:
                explore(path_entry)
        else:
            add_json_files(path)

    explore(root)

    return imgs

# todo: Dataloader NDDS
class MultipleVertexJson(data.Dataset):
    """
    Dataloader for the data generated by NDDS (https://github.com/NVIDIA/Dataset_Synthesizer).
    This is the same data as the data used in FAT.
    """

    def __init__(self, root, transform=None, nb_vertex=8,
                 keep_orientation=True,
                 normal=None, test=False,
                 target_transform=None,
                 loader=default_loader,
                 objectsofinterest="",
                 img_size=400,
                 save=False,
                 noise=2,
                 data_size=None,
                 sigma=16,
                 random_translation=(25.0, 25.0),
                 random_rotation=15.0,
                 train = None,
                 ):
        ###################
        self.save = save
        self.objectsofinterest = objectsofinterest
        self.img_size = img_size
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.imgs = []
        self.test = test
        self.normal = normal
        self.keep_orientation = keep_orientation
        self.save = save
        self.noise = noise
        self.data_size = data_size
        self.sigma = sigma
        self.random_translation = random_translation
        self.random_rotation = random_rotation

        #train test
        self.train = train

        def load_data(path):
            '''Recursively load the data.  This is useful to load all of the FAT dataset.'''
            imgs = loadimages(path)

            # Check all the folders in path
            for name in os.listdir(str(path)):
                imgs += loadimages(path + "/" + name)
            return imgs

        self.imgs = load_data(root)

        # Shuffle the data, this is useful when we want to use a subset.
        # np.random.shuffle(self.imgs)
        # Deactivated for the metric,
        # Damit pose_transform und die predicted pose zusammen passen ...

    def __len__(self):
        # When limiting the number of data
        if not self.data_size is None:
            return int(self.data_size)

        return len(self.imgs)

    def __getitem__(self, index):
        """
        Depending on how the data loader is configured,
        this will return the debug info with the cuboid drawn on it,
        this happens when self.save is set to true.
        Otherwise, during training this function returns the
        belief maps and affinity fields and image as tensors.
        """
        path, name, txt = self.imgs[index]
        img = self.loader(path)

        img_size = img.size
        img_size = (400, 400) #https://github.com/NVlabs/Deep_Object_Pose/issues/16 Image size to train is 400x400  then it can be oused for everything, in dope w640xh480

        loader = loadjson

        data = loader(txt, self.objectsofinterest, img)

        #fuer die metric
        pose_transform = data['pose_transform']

        # projected_cuboid = data['projected_cuboid']
        # cuboid_gt = data['cuboid']
        # todo: pose_gu? pose_GT?
        #list to tensor
        #print("MultipleVertexJson:pose_transform: ", pose_transform)


        pointsBelief = data['pointsBelief']
        objects_centroid = data['centroids']
        points_all = data['points']
        points_keypoints = data['keypoints_2d']
        translations = torch.from_numpy(np.array(
            data['translations'])).float()
        rotations = torch.from_numpy(np.array(
            data['rotations'])).float()


        if len(points_all) == 0:
            points_all = torch.zeros(1, 10, 2).double()

        # self.save == true assumes there is only
        # one object instance in the scene.
        if translations.size()[0] > 1:
            translations = translations[0].unsqueeze(0)
            rotations = rotations[0].unsqueeze(0)

        # If there are no objects, still need to return similar shape array
        if len(translations) == 0:
            translations = torch.zeros(1, 3).float()
            rotations = torch.zeros(1, 4).float()

        # Camera intrinsics
        path_cam = path.replace(name, '_camera_settings.json')
        with open(path_cam) as data_file:
            data = json.load(data_file)
        # Assumes one camera
        cam = data['camera_settings'][0]['intrinsic_settings']

        matrix_camera = np.zeros((3, 3))
        matrix_camera[0, 0] = cam['fx']
        matrix_camera[1, 1] = cam['fy']
        matrix_camera[0, 2] = cam['cx']
        matrix_camera[1, 2] = cam['cy']
        matrix_camera[2, 2] = 1

        # load ground thruth! ------------------------------------------------------------------------------------------
        # Load the cuboid sizes
        path_set = path.replace(name, '_object_settings.json')
        with open(path_set) as data_file:
            data = json.load(data_file)

        cuboid = torch.zeros(1)

        if self.objectsofinterest is None:
            cuboid = np.array(data['exported_objects'][0]['cuboid_dimensions'])
        else:
            for info in data["exported_objects"]:
                if self.objectsofinterest in info['class']:
                    cuboid = np.array(info['cuboid_dimensions'])

        img_original = img.copy()

        def Reproject(points, tm, rm):
            """
            Reprojection of points when rotating the image (after Data Augmetnation)
            """
            proj_cuboid = np.array(points)

            rmat = np.identity(3)
            rmat[0:2] = rm
            tmat = np.identity(3)
            tmat[0:2] = tm

            new_cuboid = np.matmul(
                rmat, np.vstack((proj_cuboid.T, np.ones(len(points)))))
            new_cuboid = np.matmul(tmat, new_cuboid)
            new_cuboid = new_cuboid[0:2].T

            return new_cuboid

        # Random image manipulation, rotation and translation with zero padding
        # also called Data Augmentation
        dx = round(np.random.normal(0, 2) * float(self.random_translation[0]))
        dy = round(np.random.normal(0, 2) * float(self.random_translation[1]))
        angle = round(np.random.normal(0, 1) * float(self.random_rotation))

        tm = np.float32([[1, 0, dx], [0, 1, dy]])
        rm = cv2.getRotationMatrix2D(
            (img.size[0] / 2, img.size[1] / 2), angle, 1)

        # Data Augmentation for Pose Transform
        # Wenn die Pose_Transform also die Ground truth hierbei
        # auch mit TM und RM transformiert und rotiert werden wuerde,
        # koennte die pose_transform mit dem bearbeiteten Bild in "img"
        # in der ADDErrorCuboid Methode berechnet werden.
        # new_pose_transform = Reproject(pose_transform, tm, rm)

        for i_objects in range(len(pointsBelief)):
            points = pointsBelief[i_objects]
            new_cuboid = Reproject(points, tm, rm)
            pointsBelief[i_objects] = new_cuboid.tolist()
            objects_centroid[i_objects] = tuple(new_cuboid.tolist()[-1])
            pointsBelief[i_objects] = list(map(tuple, pointsBelief[i_objects]))

        for i_objects in range(len(points_keypoints)):
            points = points_keypoints[i_objects]
            new_cuboid = Reproject(points, tm, rm)
            points_keypoints[i_objects] = new_cuboid.tolist()
            points_keypoints[i_objects] = list(map(tuple, points_keypoints[i_objects]))

        image_r = cv2.warpAffine(np.array(img), rm, img.size)
        result = cv2.warpAffine(image_r, tm, img.size)
        img = Image.fromarray(result)

        # Note:  All point coordinates are in the image space, e.g., pixel value.
        # This is used when we do saving --- helpful for debugging -----------------------------------------------------
        if self.save or self.test:
            # Use the save to debug the data
            if self.test:
                draw = ImageDraw.Draw(img_original)
            else:
                draw = ImageDraw.Draw(img)

            # PIL drawing functions, here for sharing draw
            def DrawKeypoints(points):
                for key in points:
                    DrawDot(key, (12, 115, 170), 7)

            def DrawLine(point1, point2, lineColor, lineWidth):
                if not point1 is None and not point2 is None:
                    draw.line([point1, point2], fill=lineColor, width=lineWidth)

            def DrawDot(point, pointColor, pointRadius):
                if not point is None:
                    xy = [point[0] - pointRadius, point[1] - pointRadius, point[0] + pointRadius,
                          point[1] + pointRadius]
                    draw.ellipse(xy, fill=pointColor, outline=pointColor)

            def DrawCube(points, which_color=0, color=None):
                '''Draw cube with a thick solid line across the front top edge.'''
                lineWidthForDrawing = 2
                lineColor1 = (255, 215, 0)  # yellow-ish
                lineColor2 = (12, 115, 170)  # blue-ish
                lineColor3 = (45, 195, 35)  # green-ish
                if which_color == 3:
                    lineColor = lineColor3
                else:
                    lineColor = lineColor1

                if not color is None:
                    lineColor = color

                # draw front
                DrawLine(points[0], points[1], lineColor, 8)  # lineWidthForDrawing)
                DrawLine(points[1], points[2], lineColor, lineWidthForDrawing)
                DrawLine(points[3], points[2], lineColor, lineWidthForDrawing)
                DrawLine(points[3], points[0], lineColor, lineWidthForDrawing)

                # draw back
                DrawLine(points[4], points[5], lineColor, lineWidthForDrawing)
                DrawLine(points[6], points[5], lineColor, lineWidthForDrawing)
                DrawLine(points[6], points[7], lineColor, lineWidthForDrawing)
                DrawLine(points[4], points[7], lineColor, lineWidthForDrawing)

                # draw sides
                DrawLine(points[0], points[4], lineColor, lineWidthForDrawing)
                DrawLine(points[7], points[3], lineColor, lineWidthForDrawing)
                DrawLine(points[5], points[1], lineColor, lineWidthForDrawing)
                DrawLine(points[2], points[6], lineColor, lineWidthForDrawing)

                # draw dots
                DrawDot(points[0], pointColor=(255, 255, 255), pointRadius=3)
                DrawDot(points[1], pointColor=(0, 0, 0), pointRadius=3)

            # Draw all the found objects.
            for points_belief_objects in pointsBelief:
                DrawCube(points_belief_objects)
            for keypoint in points_keypoints:
                DrawKeypoints(keypoint)

            img = self.transform(img)

            return {
                "img": img,
                "translations": translations,
                "rot_quaternions": rotations,
                'pointsBelief': np.array(points_all[0]),
                'matrix_camera': matrix_camera,
                'img_original': np.array(img_original),
                'cuboid': cuboid,
                'file_name': name,
            }

        # Create the belief map
        beliefsImg = CreateBeliefMap(
            img,
            pointsBelief=pointsBelief,
            nbpoints=9,
            sigma=self.sigma)

        # Create the image maps for belief
        transform = transforms.Compose([transforms.Resize(min(img_size))])
        totensor = transforms.Compose([transforms.ToTensor()])

        for j in range(len(beliefsImg)):
            beliefsImg[j] = self.target_transform(beliefsImg[j])
            # beliefsImg[j].save('{}.png'.format(j))
            beliefsImg[j] = totensor(beliefsImg[j])

        beliefs = torch.zeros((len(beliefsImg), beliefsImg[0].size(1), beliefsImg[0].size(2)))
        for j in range(len(beliefsImg)):
            beliefs[j] = beliefsImg[j][0]

        # Create affinity maps
        scale = 8
        if min(img.size) / 8.0 != min(img_size) / 8.0:
            # print (scale)
            scale = min(img.size) / (min(img_size) / 8.0)

        affinities = GenerateMapAffinity(img, 8, pointsBelief, objects_centroid, scale)
        img = self.transform(img)

        # Transform the images for training input
        w_crop = np.random.randint(0, img.size[0] - img_size[0] + 1)
        h_crop = np.random.randint(0, img.size[1] - img_size[1] + 1)
        transform = transforms.Compose([transforms.Resize(min(img_size))])
        totensor = transforms.Compose([transforms.ToTensor()])

        if not self.normal is None:
            normalize = transforms.Compose([transforms.Normalize
                                            ((self.normal[0], self.normal[0], self.normal[0]),
                                             (self.normal[1], self.normal[1], self.normal[1])),
                                            AddNoise(self.noise)])
        else:
            normalize = transforms.Compose([AddNoise(0.0001)])

        img = crop(img, h_crop, w_crop, img_size[1], img_size[0])
        img = totensor(img)

        img = normalize(img)

        w_crop = int(w_crop / 8)
        h_crop = int(h_crop / 8)

        affinities = affinities[:, h_crop:h_crop + int(img_size[1] / 8), w_crop:w_crop + int(img_size[0] / 8)]
        beliefs = beliefs[:, h_crop:h_crop + int(img_size[1] / 8), w_crop:w_crop + int(img_size[0] / 8)]

        if affinities.size()[1] == 49 and not self.test:
            affinities = torch.cat([affinities, torch.zeros(16, 1, 50)], dim=1)

        if affinities.size()[2] == 49 and not self.test:
            affinities = torch.cat([affinities, torch.zeros(16, 50, 1)], dim=2)

        # For Metric
        transform456 = transforms.Compose([
            transforms.Resize(opt.imagesize),
            transforms.ToTensor()])
        img_original = transform456(img_original)

        return {
            'img': img,
            'img_original': img_original,
            "affinities": affinities,
            'beliefs': beliefs,
            'cuboid': cuboid,
            'matrix_camera': matrix_camera,
            'pose_transform': pose_transform,
        }
#-----------------------------------------------------------------------------------------------------------------------
# Some simple vector math functions to find the angle ------------------------------------------------------------------
# between two points, used by affinity fields.  ------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


def length(v):
    return sqrt(v[0] ** 2 + v[1] ** 2)


def dot_product(v, w):
    return v[0] * w[0] + v[1] * w[1]


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def determinant(v, w):
    return v[0] * w[1] - v[1] * w[0]


def inner_angle(v, w):
    cosx = dot_product(v, w) / (length(v) * length(w))
    rad = acos(cosx)  # in radians
    return rad * 180 / pi  # returns degrees


def py_ang(A, B=(1, 0)):
    inner = inner_angle(A, B)
    det = determinant(A, B)
    if det < 0:  # this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else:  # if the det > 0 then A is immediately clockwise of B
        return 360 - inner


def GenerateMapAffinity(img, nb_vertex, pointsInterest, objects_centroid, scale):
    """
    Function to create the affinity maps,
    e.g., vector maps pointing toward the object center.

    Args:
        img: PIL image
        nb_vertex: (int) number of points
        pointsInterest: list of points
        objects_centroid: (x,y) centroids for the obects
        scale: (float) by how much you need to scale down the image
    return:
        return a list of tensors for each point except centroid point
    """

    # Apply the downscale right now, so the vectors are correct.
    img_affinity = Image.new(img.mode, (int(img.size[0] / scale), int(img.size[1] / scale)), "black")
    # Create the empty tensors
    totensor = transforms.Compose([transforms.ToTensor()])

    affinities = []
    for i_points in range(nb_vertex):
        affinities.append(torch.zeros(2, int(img.size[1] / scale), int(img.size[0] / scale)))

    for i_pointsImage in range(len(pointsInterest)):
        pointsImage = pointsInterest[i_pointsImage]
        center = objects_centroid[i_pointsImage]
        for i_points in range(nb_vertex):
            point = pointsImage[i_points]
            affinity_pair, img_affinity = getAfinityCenter(int(img.size[0] / scale),
                                                           int(img.size[1] / scale),
                                                           tuple((np.array(pointsImage[i_points]) / scale).tolist()),
                                                           tuple((np.array(center) / scale).tolist()),
                                                           img_affinity=img_affinity, radius=1)

            affinities[i_points] = (affinities[i_points] + affinity_pair) / 2

            # Normalizing
            v = affinities[i_points].numpy()

            xvec = v[0]
            yvec = v[1]

            norms = np.sqrt(xvec * xvec + yvec * yvec)
            nonzero = norms > 0

            xvec[nonzero] /= norms[nonzero]
            yvec[nonzero] /= norms[nonzero]

            affinities[i_points] = torch.from_numpy(np.concatenate([[xvec], [yvec]]))
    affinities = torch.cat(affinities, 0)

    return affinities


def getAfinityCenter(width, height, point, center, radius=7, img_affinity=None):
    """
    Function to create the affinity maps,
    e.g., vector maps pointing toward the object center.

    Args:
        width: image wight
        height: image height
        point: (x,y)
        center: (x,y)
        radius: pixel radius
        img_affinity: tensor to add to
    return:
        return a tensor
    """
    tensor = torch.zeros(2, height, width).float()

    # Create the canvas for the afinity output
    imgAffinity = Image.new("RGB", (width, height), "black")
    totensor = transforms.Compose([transforms.ToTensor()])

    draw = ImageDraw.Draw(imgAffinity)
    r1 = radius
    p = point
    draw.ellipse((p[0] - r1, p[1] - r1, p[0] + r1, p[1] + r1), (255, 255, 255))

    del draw

    # Compute the array to add the afinity
    array = (np.array(imgAffinity) / 255)[:, :, 0]

    angle_vector = np.array(center) - np.array(point)
    angle_vector = normalize(angle_vector)
    affinity = np.concatenate([[array * angle_vector[0]], [array * angle_vector[1]]])

    # print (tensor)
    if not img_affinity is None:
        # Find the angle vector
        # print (angle_vector)
        if length(angle_vector) > 0:
            angle = py_ang(angle_vector)
        else:
            angle = 0
        # print(angle)
        c = np.array(colorsys.hsv_to_rgb(angle / 360, 1, 1)) * 255
        draw = ImageDraw.Draw(img_affinity)
        draw.ellipse((p[0] - r1, p[1] - r1, p[0] + r1, p[1] + r1), fill=(int(c[0]), int(c[1]), int(c[2])))
        del draw
    re = torch.from_numpy(affinity).float() + tensor
    return re, img_affinity


def CreateBeliefMap(img, pointsBelief, nbpoints, sigma=16):
    """
    Args:
        img: image
        pointsBelief: list of points in the form of
                      [nb object, nb points, 2 (x,y)]
        nbpoints: (int) number of points, DOPE uses 8 points here
        sigma: (int) size of the belief map point
    return:
        return an array of PIL black and white images representing the
        belief maps
    """
    beliefsImg = []
    sigma = int(sigma)
    for numb_point in range(nbpoints):
        array = np.zeros(img.size)
        out = np.zeros(img.size)

        for point in pointsBelief:
            p = point[numb_point]
            w = int(sigma * 2)
            if p[0] - w >= 0 and p[0] + w < img.size[0] and p[1] - w >= 0 and p[1] + w < img.size[1]:
                for i in range(int(p[0]) - w, int(p[0]) + w):
                    for j in range(int(p[1]) - w, int(p[1]) + w):
                        array[i, j] = np.exp(-(((i - p[0]) ** 2 + (j - p[1]) ** 2) / (2 * (sigma ** 2))))

        stack = np.stack([array, array, array], axis=0).transpose(2, 1, 0)
        imgBelief = Image.new(img.mode, img.size, "black")
        beliefsImg.append(Image.fromarray((stack * 255).astype('uint8')))
    return beliefsImg


def crop(img, i, j, h, w):
    """
    Crop the given PIL.Image.

    Args:
        img (PIL.Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL.Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))


class AddRandomContrast(object):
    """
    Apply some random contrast from PIL
    """

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, im):
        contrast = ImageEnhance.Contrast(im)
        im = contrast.enhance(np.random.normal(1, self.sigma))
        return im


class AddRandomBrightness(object):
    """
    Apply some random brightness from PIL
    """

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, im):
        bright = ImageEnhance.Brightness(im)
        im = bright.enhance(np.random.normal(1, self.sigma))
        return im


class AddNoise(object):
    """
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        # t = torch.FloatTensor(tensor.size()).uniform_(self.min,self.max)
        t = torch.FloatTensor(tensor.size()).normal_(0, self.std)

        t = tensor.add(t)
        t = torch.clamp(t, -1, 1)  # this is expansive
        return t


irange = range


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize == True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each == True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=4, padding=2, mean=None, std=None):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    from PIL import Image

    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=10, pad_value=1)
    if not mean is None:
        ndarr = grid.mul(std).add(mean).mul(255).byte().transpose(0, 2).transpose(0, 1).numpy()
    else:
        ndarr = grid.mul(0.5).add(0.5).mul(255).byte().transpose(0, 2).transpose(0, 1).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def DrawLine(point1, point2, lineColor, lineWidth, draw):
    if not point1 is None and not point2 is None:
        draw.line([point1, point2], fill=lineColor, width=lineWidth)


def DrawDot(point, pointColor, pointRadius, draw):
    if not point is None:
        xy = [point[0] - pointRadius, point[1] - pointRadius, point[0] + pointRadius, point[1] + pointRadius]
        draw.ellipse(xy, fill=pointColor, outline=pointColor)


def DrawCube(points, which_color=0, color=None, draw=None):
    '''Draw cube with a thick solid line across the front top edge.'''
    lineWidthForDrawing = 2
    lineWidthThick = 8
    lineColor1 = (255, 215, 0)  # yellow-ish
    lineColor2 = (12, 115, 170)  # blue-ish
    lineColor3 = (45, 195, 35)  # green-ish
    if which_color == 3:
        lineColor = lineColor3
    else:
        lineColor = lineColor1

    if not color is None:
        lineColor = color

    # draw front
    DrawLine(points[0], points[1], lineColor, lineWidthThick, draw)
    DrawLine(points[1], points[2], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[3], points[2], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[3], points[0], lineColor, lineWidthForDrawing, draw)

    # draw back
    DrawLine(points[4], points[5], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[6], points[5], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[6], points[7], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[4], points[7], lineColor, lineWidthForDrawing, draw)

    # draw sides
    DrawLine(points[0], points[4], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[7], points[3], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[5], points[1], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[2], points[6], lineColor, lineWidthForDrawing, draw)

    # draw dots
    DrawDot(points[0], pointColor=lineColor, pointRadius=4, draw=draw)
    DrawDot(points[1], pointColor=lineColor, pointRadius=4, draw=draw)

#-----------------------------------------------------------------------------------------------------------------------
# functions to show an image
#-----------------------------------------------------------------------------------------------------------------------
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

    #plt.imshow(transforms.ToPILImage()(img), interpolation="bicubic")

#-----------------------------------------------------------------------------------------------------------------------
# TRAINING CODE MAIN STARTING HERE -------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

start_time = datetime.datetime.now().time()
print ("start:" , start_time)

conf_parser = argparse.ArgumentParser(
    description=__doc__, # printed with -h/--help
    # Don't mess with format of description
    formatter_class=argparse.RawDescriptionHelpFormatter,
    # Turn off help, so we print all options in response to -h
    add_help=False
    )
conf_parser.add_argument("-c", "--config",
                        help="Specify config file", metavar="FILE")

parser = argparse.ArgumentParser()

parser.add_argument('--data',
    default = "",
    help='path to training data')

parser.add_argument('--datatest',
    default="",
    help='path to data testing set')

parser.add_argument('--object',
    default=None,
    help='In the dataset which objet of interest')

parser.add_argument('--workers',
    type=int,
    default=8,
    help='number of data loading workers')

parser.add_argument('--batchsize',
    type=int,
    default=8, #TODO: default was 32 i will decrease to 16 because of the meomry
    help='input batch size')

parser.add_argument('--imagesize',
    type=int,
    default=400,
    help='the height / width of the input image to network')

parser.add_argument('--lr',
    type=float,
    default=0.0001,
    help='learning rate, default=0.001')

parser.add_argument('--noise',
    type=float,
    default=2.0,
    help='gaussian noise added to the image')

parser.add_argument('--net',
    default='',
    help="path to net (to continue training)")

parser.add_argument('--namefile',
    default='epoch',
    help="name to put on the file of the save weights")

parser.add_argument('--manualseed',
    type=int,
    help='manual seed')

parser.add_argument('--epochs',
    type=int,
    default=60,
    help="number of epochs to train")

parser.add_argument('--loginterval',
    type=int,
    default=100)

parser.add_argument('--gpuids',
    nargs='+',
    type=int,
    default=[0],
    help='GPUs to use')

parser.add_argument('--outf',
    default='tmp',
    help='folder to output images and model checkpoints, it will \
    add a train_ in front of the name')

parser.add_argument('--sigma',
    default=4,
    help='keypoint creation size for sigma')

parser.add_argument('--save',
    action="store_true",
    help='save a visual batch and quit, this is for\
    debugging purposes')

parser.add_argument("--pretrained",
    default=True,
    help='do you want to use vgg imagenet pretrained weights')

parser.add_argument('--nbupdates',
    default=None,
    help='nb max update to network, overwrites the epoch number\
    otherwise uses the number of epochs')

parser.add_argument('--datasize',
    default=None,
    help='randomly sample that number of entries in the dataset folder')

# Read the config but do not overwrite the args written
args, remaining_argv = conf_parser.parse_known_args()
defaults = { "option":"default" }

if args.config:
    config = ConfigParser.SafeConfigParser()
    config.read([args.config])
    defaults.update(dict(config.items("defaults")))

parser.set_defaults(**defaults)
parser.add_argument("--option")

opt = parser.parse_args(remaining_argv)

if opt.pretrained in ['false', 'False']:
	opt.pretrained = False

#-----------------------------------------------------------------------------------------------------------------------
# OPT SAVE IMAGES? -----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
if not "/" in opt.outf:
    opt.outf = "train_{}".format(opt.outf)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualseed is None:
    opt.manualseed = random.randint(1, 10000)

# save the hyper parameters passed
with open (opt.outf + '/header.txt','w') as file:
    file.write(str(opt)+"\n")

with open (opt.outf + '/header.txt','w') as file:
    file.write(str(opt))
    file.write("seed: " + str(opt.manualseed)+'\n')

#print ("start:" , start_time)
with open (opt.outf+'/header.txt','a') as file:
    file.write("\nstart: " + str(start_time)+"\n")

# set the manual seed.
random.seed(opt.manualseed)
torch.manual_seed(opt.manualseed)
torch.cuda.manual_seed_all(opt.manualseed)

# save
if not opt.save:
    contrast = 0.2
    brightness = 0.2
    noise = 0.1
    normal_imgs = [0.59,0.25]
    transform = transforms.Compose([
                               AddRandomContrast(contrast),
                               AddRandomBrightness(brightness),
                               transforms.Scale(opt.imagesize),
                               ])
else:
    contrast = 0.00001
    brightness = 0.00001
    noise = 0.00001
    normal_imgs = None
    transform = transforms.Compose([
                           transforms.Resize(opt.imagesize),
                           transforms.ToTensor()])

print ("load data")
#load the dataset using the loader in utils_pose
trainingdata = None
if not opt.data == "":
    train_dataset = MultipleVertexJson(
        root = opt.data,
        objectsofinterest=opt.object,
        keep_orientation = True,
        noise = opt.noise,
        sigma = opt.sigma,
        data_size = opt.datasize,
        save = opt.save,
        transform = transform,
        normal = normal_imgs,
        target_transform = transforms.Compose([
                               transforms.Scale(opt.imagesize//8),
            ]),
        train = True,
        )
    trainingdata = torch.utils.data.DataLoader(train_dataset,
        batch_size = opt.batchsize,
        shuffle = True,
        num_workers = opt.workers,
        pin_memory = True
        )

if opt.save:
    for i in range(2):
        images = iter(trainingdata).next()
        if normal_imgs is None:
            normal_imgs = [0,1]
        save_image(images['img'],'{}/train_{}.png'.format( opt.outf,str(i).zfill(5)),mean=normal_imgs[0],std=normal_imgs[1])

        print (i)

    print ('things are saved in {}'.format(opt.outf))
    quit()

testingdata = None
if not opt.datatest == "":
    testingdata = torch.utils.data.DataLoader(
        MultipleVertexJson(
            root = opt.datatest,
            objectsofinterest=opt.object,
            keep_orientation = True,
            noise = opt.noise,
            sigma = opt.sigma,
            data_size = opt.datasize,
            save = opt.save,
            transform = transform,
            normal = normal_imgs,
            target_transform = transforms.Compose([
                                   transforms.Scale(opt.imagesize//8),
                ]),
            train = False
            ),
        batch_size = opt.batchsize,
        shuffle = True,
        num_workers = opt.workers,
        pin_memory = True)

if not trainingdata is None:
    print('training data: {} batches'.format(len(trainingdata)))
if not testingdata is None:
    print ("testing data: {} batches".format(len(testingdata)))
print('load models')


#-----------------------------------------------------------------------------------------------------------------------
# Finetuning -----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

##################################################
#Load Model
#cracker_60_Kopie.pth model
# Initialize model
##################################################

#print ('test')
#print (torch.version.cuda)

net = DopeNetwork(pretrained=opt.pretrained).cuda()
net = torch.nn.DataParallel(net,device_ids=opt.gpuids).cuda()
print ('Load net')

net.load_state_dict(torch.load('cracker_60_Kopie.pth'))
with open (opt.outf+'/header.txt','a') as file:
    file.write("\nFine-tuned with: cracker_60_Kopie.pth\n")

# Model class must be defined somewhere
# funktioniert wohl nicht weil es voher geladen werden muss weil hier nicht das gesamte model gespeichert wurde sondern nur die values gespeichert sind
#model = torch.load('cracker_60_Kopie.pth')

# Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
# Failing to do this will yield inconsistent inference results.
#print ('Evaluate Model')
#net.eval()

#print ('Modell:')
#print (model)

# Freeze vgg model weights
#for param in net.module.vgg.parameters():
    #param.requires_grad = False
# Freeze model weights
#for param in net.module.m1_2.parameters():
#    param.requires_grad = False
#for param in net.module.m2_2.parameters():
#    param.requires_grad = False
#for param in net.module.m3_2.parameters():
#    param.requires_grad = False
#for param in net.module.m4_2.parameters():
#    param.requires_grad = False
#for param in net.module.m5_2.parameters():
#    param.requires_grad = False
#for param in net.module.m6_2.parameters():
#    param.requires_grad = False

# Freeze model weights
for param in net.module.m1_1.parameters():
    param.requires_grad = False
for param in net.module.m2_1.parameters():
    param.requires_grad = False
for param in net.module.m3_1.parameters():
    param.requires_grad = False
for param in net.module.m4_1.parameters():
    param.requires_grad = False
for param in net.module.m5_1.parameters():
    param.requires_grad = False
for param in net.module.m6_1.parameters():
    param.requires_grad = False

with open (opt.outf+'/header.txt','a') as file:
    file.write("\nTrained on vgg and m*_2 layers\nLayer m*_1 eingrforen")

#for param in net.parameters():
#   print("param: ", param.requires_grad)

#quit()

#How to remove the last FC layer from a ResNet model in PyTorch?
#https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch

#print ('Moell neu2:')
#print (net)

# Remove last x layer
#my_model1123 = nn.Sequential(*list(net.module.m6_1)[:-13])#5 # strips off last x layer's
#net.module.m3_1 = None
#net.module.m4_1 = None
#net.module.m5_1 = None
#net.module.m6_1 = None #Remove complete Stage


#list(net.modules()) # to inspect the modules of your model
#print ('Modell after deleted fully convolutional:')
#print (net.module)

#with open (opt.outf+'/header.txt','a') as file:
#    file.write("\nModel after deleted fully convolutional: " + str(net.module)+"\n")


#Fine-tune the last LAyer
# i = 13 oder 14?
numBeliefMap=9
numAffinity=16
stop_at_stage=6

global deleted_layers
deleted_layers = 26

#self.m1_1 = DopeNetwork.create_stage(128, numAffinity, True)
#self.m2_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
#                                             numAffinity, False)

my_model1123_m3_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
my_model1123_m4_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
my_model1123_m5_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
my_model1123 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)

#4 Layer
#my_model1123.add_module(str(6),nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3))

#3 Layer
#my_model1123.add_module(str(7), nn.ReLU(inplace=True))
#my_model1123.add_module(str(8),nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3))

#2 Layer
#my_model1123.add_module(str(9), nn.ReLU(inplace=True))
#my_model1123.add_module(str(10), nn.Conv2d(128, 128, kernel_size=1, stride=1))

#1 Last convolution
#my_model1123.add_module(str(11), nn.ReLU(inplace=True))
#my_model1123.add_module(str(12), nn.Conv2d(128, 16, kernel_size=1, stride=1))

#128 problem muss zu 16 werden? aufgrund von
# RuntimeError: The size of tensor a (128) must match the size of tensor b (16) at non-singleton dimension 1
# in line loss_tmp = ((l - target_affinity) * (l - target_affinity)).mean()


#copy renewed stage to DOPE

#net.module.m3_1 = my_model1123_m3_1
#net.module.m4_1 = my_model1123_m4_1
#net.module.m5_1 = my_model1123_m5_1
#net.module.m6_1 = my_model1123

#for param in net.parameters():
#    print("param: ", param.requires_grad)
#quit()

net.cuda() # this or the other one?
#net = torch.nn.DataParallel(net,device_ids=opt.gpuids).cuda() #Push model to Cuda?

#print('Modell neu:')
#print(net)
#with open (opt.outf+'/header.txt','a') as file:
#    file.write("\nModell neu: " + str(net)+"\n")


print ('finished Fine-tuning configuration')

#python train.py --data path/to/FAT --object soup --outf soup
#--gpuids 0 1 2 3 4 5 6 7

# Error:
# Traceback (most recent call last):
#   File "test2.py", line 111, in <module>
#     g_path2package = rospack.get_path('dope')
#   File "/home/nils/catkin_ws/local/lib/python2.7/site-packages/rospkg/rospack.py", line 203, in get_path
#     raise ResourceNotFound(name, ros_paths=self._ros_paths)
# rospkg.common.ResourceNotFound: dope
# ROS path [0]=/opt/ros/kinetic/share/ros
# ROS path [1]=/opt/ros/kinetic/share

#Solution: source devel/setup.bash


# source devel/setup.bash
# source /home/nils/catkin_ws/bin/activate

# Training command:
# python test.py --data /media/nils/Ubuntu-TMP/Stereo_Dataset_Luedeke/single/CandyShop/?? --datatest /media/nils/Ubuntu-TMP/Stereo_Dataset_Luedeke/single/CandyShop/Validation  --object CandyShop2 --outf CandyShop2

# null test: python test2.py --object test --outf test

# python test.py --data /media/nils/Ubuntu-TMP/Stereo_Dataset_Luedeke/single/CandyShop/?? --datatest /media/nils/Ubuntu-TMP/Stereo_Dataset_Luedeke/single/CandyShop/Validation  --object CandyShop2 --outf CandyShop2

# python test2.py --data /media/nils/Ubuntu-TMP/CandyShop_neu/CandyShop_Overfitting_test/ --datatest /media/nils/Ubuntu-TMP/Stereo_Dataset_Luedeke/single/CandyShop/Validation  --object CandyShop2 --outf CandyShop2

# python test2.py --data /media/nils/Ubuntu-TMP/fat/single/003_cracker_box_16k/kitch/ --datatest /media/nils/Ubuntu-TMP/fat/single/003_cracker_box_16k/kitch  --object 003_cracker_box_16k --outf 003_cracker_box_16k

# python test2.py --data /media/nils/Ubuntu-TMP/fat/single/003_cracker_box_16k/asdf/ --datatest /media/nils/Ubuntu-TMP/fat/single/003_cracker_box_16k/kitedemo_4/ --object 003_cracker_box_16k --outf 003_cracker_box_16k

# python test2.py --data /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/FerreroKuesschen/Overfitting --datatest /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/FerreroKuesschen/Overfitting --object FerreroKuesschen --outf FerreroKuesschen


# save
# python test2.py --data /media/nils/Ubuntu-TMP/CandyShop_neu/CandyShop_Overfitting_test/ --datatest /media/nils/Ubuntu-TMP/Stereo_Dataset_Luedeke/single/CandyShop/Validation  --object CandyShop2 --outf CandyShop2 --save



# Cogsys rechner
# overfitting datensatz
# python train.py --data /home/luedeke/Stereo_Dataset_neu/Single/CandyShop/CandyShop_Overfitting_3000 --datatest /home/luedeke/Stereo_Dataset/Single/CandyShop/Validation --object CandyShop2 --outf CandyShop2

# python train.py --data /home/luedeke/Stereo_Dataset_neu/Single/CandyShop/Livingroom --datatest /home/luedeke/Stereo_Dataset_neu/Single/CandyShop/Validation --object CandyShop2 --outf CandyShop2


# grossser datensatz
# python test.py --data /home/luedeke/Stereo_Dataset/Single/CandyShop/RoomAndBerlin --datatest /home/luedeke/Stereo_Dataset/Single/CandyShop/Validation --object CandyShop2 --outf CandyShop2

# kleiner datensatz
# python test.py --data /home/luedeke/Stereo_Dataset/Single/CandyShop/RoomAndBerlin/Room --datatest /home/luedeke/Stereo_Dataset/Single/CandyShop/Validation   --object CandyShop2 --outf CandyShop2


# python test.py --data /media/nils/Ubuntu-TMP/Stereo_Dataset_Luedeke/ --object FerreroKuesschen --outf FerreroKuesschen
# rsync -avz /media/nils/Ubuntu-TMP/Stereo_Dataset_Luedeke/single/ luedeke@dlsys-MACHINE:/home/luedeke/Stereo_Dataset/Single

# rsync -avz /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/FerreroKuesschen/ luedeke@dlsys-MACHINE:/home/luedeke/Dataset_Luedeke_Neu/

# dekstop to notebook
# ssh nils-nb@192.168.178.22
# rsync -avz /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/FerreroKuesschen/ nils-nb@192.168.178.22:/media/nils-nb/UbuntuTMP/FerreroKuesschen/
# rsync -avz /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/CandyShop/ nils-nb@192.168.178.22:/media/nils-nb/UbuntuTMP/CandyShop/

#-----------------------------------------------------------------------------------------------------------------------
# TRAIN ----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

#How much data are you using?
# You might want to change the learning rate as well.
# Only training on FAT for some objects might not be enough.
# The scissor is quite a small object.
# When I was developing DOPE I had create debug tools to look at the output of the neural network.
# So what I would do first would be to look at the output of the neural network on the training images to see that it actually outputs the belief maps.
# Then I would test on other images.
#https://github.com/NVlabs/Deep_Object_Pose/issues/12

print ('Start Fine-tuning Training Layers: ', str(deleted_layers))

if opt.net != '':
    print('Start Fine-tuning Training Load Torch test')
    net.load_state_dict(torch.load(opt.net))

parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.Adam(parameters, lr=opt.lr)

with open(opt.outf + '/loss_train.csv', 'w') as file:
    file.write('epoch,batchid,loss\n')

with open(opt.outf + '/loss_test.csv', 'w') as file:
    file.write('epoch,batchid,loss\n')

with open (opt.outf+'/test_metric.csv','w') as file:
    file.write("epoch,batchid,mean,detected_images,possible_images\n")

with open (opt.outf+'/train_metric.csv','w') as file:
    file.write("epoch,batchid,mean,detected_images,possible_images\n")

nb_update_network = 0
#-----------------------------------------------------------------------------------------------------------------------
# init config_pose_rosbag-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

dist_coeffs = np.zeros((4, 1))

config_name = "config_pose_rosbag.yaml"
rospack = rospkg.RosPack()
params = None
yaml_path = g_path2package + '/config/{}'.format(config_name)
with open(yaml_path, 'r') as stream:
    try:
        print("Loading DOPE parameters from '{}'...".format(yaml_path))
        params = yaml.load(stream)
        print('DOPE parameters loaded.')
    except yaml.YAMLError as exc:
        print("Error: ", exc)

if "dist_coeffs" in params["camera_settings"]:
    dist_coeffs = np.array(params["camera_settings"]['dist_coeffs'])

config_detect = 0
config_detect = lambda: None
config_detect.mask_edges = 1
config_detect.mask_faces = 1
config_detect.vertex = 1
config_detect.threshold = 0.5
config_detect.softmax = 1000
config_detect.thresh_angle = params['thresh_angle']
config_detect.thresh_map = params['thresh_map']
config_detect.sigma = params['sigma']
config_detect.thresh_points = params["thresh_points"]

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


def ADDErrorCuboid(pose_gu, pose_gt, cuboid):
    """
        T and the estimated rotation R and translation T
        we compute the average distance of all model points x from their transformed versions
            m = avg Strich(Rx + T) minus ( R x + T)Strich

        Compute the ADD error for a given cuboid.
        pose_gu is the predicted pose as a matrix
        pose_gt is the ground thruth pose as a matrix
        cuboid is a Cuboid3D object (see inference/cuboid.py)
    """
    from scipy import spatial

    #print("ADDErrorCuboid:pred_obj: ", pose_gt)
    #print("ADDErrorCuboid:pose_gu: ", pose_gu)

    # obj = self.__obj_model
    vertices = np.array(cuboid._vertices)
    #print("ADDErrorCuboid:cuboid._vertices: ", vertices)
    #print ("ADDErrorCuboid:pose_gu.shape: ", pose_gu.shape)
    #print ("ADDErrorCuboid:vertices.shape: ", vertices.shape)
    vertices = np.insert(vertices, 3, 1, axis=1)
    #print("ADDErrorCuboid:vertices:insert ", vertices)
    #print ("ADDErrorCuboid:vertices.shape: ", vertices.shape)
    vertices = np.rot90(vertices, 3)
    #print("ADDErrorCuboid:vertices:rot90 ", vertices)

    #print ("ADDErrorCuboid:pose_gu.shape: ", pose_gu.shape)
    #pose_gu = np.insert(pose_gu, 3, 1, axis=1)
    #print("ADDErrorCuboid:pose_gu: ", pose_gu)
    #pose_gu2 = np.array([pose_gu, [0, 0, 0, 1]])
    #asdF = np.array([0, 0, 0, 1])
    #pose_gu2 = np.vstack((pose_gu, asdF))
    #print ("ADDErrorCuboid:pose_gu.shape: ", pose_gu2.shape)
    #print("ADDErrorCuboid:pose_gu: ", pose_gu2)
    obj = vertices
    pred_obj = np.matmul(pose_gu, obj)

    # obj = self.__obj_model
    # pred_obj = np.matmul(pose_gu, obj)
    # print (pred_obj)

    actual_obj = np.matmul(pose_gt, obj)
    #print("ADDErrorCuboid:pose_gt.shape: ", pose_gt.shape)
    #print("ADDErrorCuboid:pose_gt: ", pose_gt)
    #print("ADDErrorCuboid:actual_obj: ", actual_obj)
    # actual_obj = np.matmul(self.LtNT, actual_obj_l)
    # print("PREDICTED OBJECT\n", pred_obj)
    # print("ACTUAL OBJECT\n", actual_obj)
    dist = spatial.distance.cdist(pred_obj.T, actual_obj.T, 'euclidean')
    #print("ADDErrorCuboid:dist: ", dist)
    true_dist = [dist[i][i] for i in range(len(dist))]
    #print("ADDErrorCuboid:true_dist: ", true_dist)
    #print("ADDErrorCuboid:np.mean: ", np.mean(true_dist))
    # for i in range(len(true_dist)):
    #    if true_dist[i] >6000:
    #        print(i, true_dist[i])
    # print (true_dist)
    # raise()
    return np.mean(true_dist)



def _runnetwork(epoch, loader, train=True):
    global nb_update_network
    # net
    if train:
        net.train()
    else:
        net.eval()

    for batch_idx, targets in enumerate(loader):
        #ValueError: No JSON object could be decoded
        # -> solved with other Dataset without Umlaute

        data = Variable(targets['img'].cuda())

        output_belief, output_affinities = net(data)

        #RuntimeError: Input    type(torch.cuda.FloatTensor) and weight    type(torch.FloatTensor)
        #should    be    the    same -> Solved with net.cuda()  in Finetuning


        if train:
            optimizer.zero_grad()
        target_belief = Variable(targets['beliefs'].cuda())
        target_affinity = Variable(targets['affinities'].cuda())

        loss = None

        # Belief maps loss
        for l in output_belief:  # output, each belief map layers.
            if loss is None:
                loss = ((l - target_belief) * (l - target_belief)).mean()
            else:
                loss_tmp = ((l - target_belief) * (l - target_belief)).mean()
                loss += loss_tmp

        # Affinities loss
        for l in output_affinities:  # output, each belief map layers.
            loss_tmp = ((l - target_affinity) * (l - target_affinity)).mean() #RuntimeError: The size of tensor a (128) must match the size of tensor b (16)
            # at non-singleton dimension 1
            loss += loss_tmp

        if train:
            loss.backward()
            optimizer.step()
            nb_update_network += 1

        if train:
            namefile = '/loss_train.csv'
        else:
            namefile = '/loss_test.csv'

        with open(opt.outf + namefile, 'a') as file:
            s = '{}, {},{:.15f}\n'.format(
                epoch, batch_idx, loss.data[0])
            # print (s)
            file.write(s)

        #print('test batch-idx:{} length data{} /data loadaer {} ({:.0f}%)]\tLoss: {:.15f}'.format(
        #    epoch, batch_idx, len(data), len(loader.dataset),
        #           100. * batch_idx / len(loader), loss.data[0]))

        if train:
            if batch_idx % opt.loginterval == 0:
                print('Train Epoch:  {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset),
                           100. * batch_idx / len(loader), loss.data[0]))

                # Compute the Metric -----------------------------------------------------------------------------------
                # print("data: ", data)
                pnp_solvers = {}
                results = {}
                results_insg = []

                matrix_camera = 0
                # matrix_camera = Variable(targets['matrix_camera'].cuda())
                # Initialize matrix_camera parameters
                matrix_camera = np.zeros((3, 3))
                matrix_camera[0, 0] = params["camera_settings"]['fx']
                matrix_camera[1, 1] = params["camera_settings"]['fy']
                matrix_camera[0, 2] = params["camera_settings"]['cx']
                matrix_camera[1, 2] = params["camera_settings"]['cy']
                matrix_camera[2, 2] = 1
                # print("matrix_camera ", matrix_camera)

                cuboid = 0
                projected_points = np.empty((0))

                for model in params['weights']:
                    # print("for model in params['weights']:model: ", model)
                    cuboid = Cuboid3d(params['dimensions'][model])
                    # print("params['dimensions'][model]: ", params['dimensions'][model])
                    # print("params['draw_colors'][model]: ", params["draw_colors"][model])
                    # init pnp_solversf
                    pnp_solvers[model] = \
                        CuboidPNPSolver(
                            model,
                            matrix_camera,
                            Cuboid3d(params['dimensions'][model]),
                            dist_coeffs=dist_coeffs
                        )

                    img_org = targets['img_original']
                    # print("img_org: ", img_org)
                    # print("img_org:type: ", type(img_org))

                    # image_tensor = transform(g_img)
                    i = 0
                    for images_tmp in img_org:
                        # print("images_tmp: ", images_tmp)
                        # print("images_tmp:len ", len(images_tmp))
                        # print("images_tmp:type: ", type(images_tmp))

                        # Detect object
                        # save_image(images_tmp, '{}/train_indetector_{}.png'.format(opt.outf, str(batch_idx + i).zfill(5)),
                        #       mean=0, std=1)

                        image_torch = Variable(images_tmp).cuda().unsqueeze(0)
                        res = ObjectDetector.detect_object_in_image_alreadytensor(
                            net,
                            pnp_solvers[model],
                            image_torch,
                            config_detect
                        )
                        # print("_runnetwork:res: ", res)
                        if res == []:
                            results_insg.append("test")
                        else:
                            results_insg.append(res)
                        i = i + 1
                    # print("_runnetwork:results_insg: ", results_insg)
                    # Find objects from network output
                    # print("_runnetwork:results: ", results)

                matrixHomogen = []
                for res in results_insg:
                    # print("for res in results_insg:")
                    if res == 'test':
                        matrixHomogen.append("test")
                        # print("asdfasfdsdf: ")
                    else:
                        # print("else")
                        for i_r, result in enumerate(res):  # enumerate(results):
                            # print("result: ", result)
                            if result["location"] is None:
                                continue
                            # print("_runnetwork:results: ", results)
                            object_name = result['name']  # welches objekt gefunden wurde
                            loc = result["location"]  # translation
                            ori = result["quaternion"]  # rotation

                            # Rx + T homogene Rigid body tramsformation
                            # quaternion zu Rotationsmatrix berechne + Transformation = Rx +T
                            quaternation = 0
                            quaternation = QtoR(ori)
                            # print("loadjson:quaternation: ", quaternation)
                            # print("loadjson:quaternation: ", type(quaternation))

                            tmp = np.insert(quaternation, 3, loc, axis=0)
                            # print("loadjson:insert: ", tmp)
                            # print("loadjson:insert:type: ", type(tmp))
                            # np.set_printoptions(precision=10)

                            resultPoseGU = np.insert(tmp, 3, [0, 0, 0, 1], axis=1)
                            matrixHomogen.append(resultPoseGU)
                            # print("loadjson:result: ", resultPoseGU)

                            # g_draw = ImageDraw.Draw(im)
                            # Draw the cube
                            # if None not in result['projected_points']:
                            #    points2d = []
                            #    for pair in result['projected_points']:
                            #        points2d.append(tuple(pair))
                            # projected_points.append(tuple(pair))

                # https://stats.stackexchange.com/questions/291820/what-is-the-definition-of-a-feature-map-aka-activation-map-in-a-convolutio
                # and feature maps??

                # pose_transform
                pose_transform = 0
                pose_transform = targets['pose_transform']
                # np.set_printoptions(precision=10)
                # pose_transform = Variable(targets['pose_transform'].cuda())
                # print("_runnetwork:pose_transform: ", pose_transform)
                # pose_transform = data['pose_transform']
                # print("_runnetwork:pose_transform: ", pose_transform)
                # print("_runnetwork:matrixHomogen: ", matrixHomogen)

                mean = 0
                i = 0  # Wie viele erkannt wurden np und fp
                image_size = len(pose_transform)  # How many Images wurden getestet

                # print("Train Metric:for tmp_pose_transform, tmp_matrixHomogen in zip(pose_transform, tmp_matrixHomogen):")
                for tmp_pose_transform, tmp_matrixHomogen in zip(pose_transform, matrixHomogen):
                    # print("Train Metric:pose_gt:pose_transform", tmp_pose_transform)
                    # print("Train Metric:pose_gu:tmp_matrixHomogen: ", tmp_matrixHomogen)
                    if tmp_matrixHomogen != 'test':
                        print("Train Metric:pose_gt:pose_transform", tmp_pose_transform)
                        print("Train Metric:pose_gu:tmp_matrixHomogen: ", tmp_matrixHomogen)
                        tmp = ADDErrorCuboid(tmp_pose_transform, tmp_matrixHomogen, cuboid)
                        print("Train Metric:ADDErrorCuboid:mean: ", tmp)
                        mean = mean + tmp
                        i = i + 1
                    # else:
                    # print("ADDErrorCuboid:mean: wrong")
                if mean != 0 and i > 1:
                    # print("ADDErrorCuboid:i: ", i)
                    mean = (mean / i)  # Durchschnitt berechnen
                    # print("ADDErrorCuboid:Durchschnitt:mean: ", mean)

                # the neural network outputs
                # belief maps to know where the cuboid object is located. Using that information and
                # the fact that we know the size and have access to the object size and the camera intrinsic
                # we find the 3d pose of the object using PnP.
                length_data = str(len(loader.dataset))
                length_data_count = len(length_data)
                test = batch_idx * len(data)
                asdf = str(test).zfill(length_data_count)
                print('Train Metric: {} [{}/{} ({:.0f}%)]\tMean: {:.15f}'.format(
                    epoch, asdf, len(loader.dataset),
                    100. * batch_idx / len(loader), mean))

                with open(opt.outf + '/train_metric.csv', 'a') as file:
                    # i = 0  # Wie viele erkannt wurden np und fp
                    # image_size = len(img_org)  # How many Images wurden getestet
                    s = '{}, {},{:.15f},{},{}\n'.format(
                        epoch, batch_idx, mean, i, image_size)
                    file.write(s)
                # Compute the Metric -----------------------------------------------------------------------------------
        else:
            if batch_idx % opt.loginterval == 0:
                length_data = str(len(loader.dataset))
                length_data_count = len(length_data)
                test = batch_idx * len(data)
                asdf = str(test).zfill(length_data_count)
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(
                    epoch, asdf, len(loader.dataset),
                    100. * batch_idx / len(loader), loss.data[0]))

                # Compute the Metric ---------------------------------------------------------------------------------------
                # print("data: ", data)
                pnp_solvers = {}
                results = {}
                results_insg = []

                matrix_camera = 0
                # matrix_camera = Variable(targets['matrix_camera'].cuda())
                # Initialize matrix_camera parameters
                matrix_camera = np.zeros((3, 3))
                matrix_camera[0, 0] = params["camera_settings"]['fx']
                matrix_camera[1, 1] = params["camera_settings"]['fy']
                matrix_camera[0, 2] = params["camera_settings"]['cx']
                matrix_camera[1, 2] = params["camera_settings"]['cy']
                matrix_camera[2, 2] = 1
                # print("matrix_camera ", matrix_camera)

                cuboid = 0
                projected_points = np.empty((0))

                for model in params['weights']:
                    # print("for model in params['weights']:model: ", model)
                    cuboid = Cuboid3d(params['dimensions'][model])
                    # print("params['dimensions'][model]: ", params['dimensions'][model])
                    # print("params['draw_colors'][model]: ", params["draw_colors"][model])
                    # init pnp_solversf
                    pnp_solvers[model] = \
                        CuboidPNPSolver(
                            model,
                            matrix_camera,
                            Cuboid3d(params['dimensions'][model]),
                            dist_coeffs=dist_coeffs
                        )

                    img_org = targets['img_original']
                    # print("img_org: ", img_org)
                    # print("img_org:type: ", type(img_org))

                    # image_tensor = transform(g_img)
                    i = 0
                    for images_tmp in img_org:
                        # print("images_tmp: ", images_tmp)
                        # print("images_tmp:len ", len(images_tmp))
                        # print("images_tmp:type: ", type(images_tmp))

                        # Detect object
                        # save_image(images_tmp, '{}/train_indetector_{}.png'.format(opt.outf, str(batch_idx + i).zfill(5)),
                        #       mean=0, std=1)

                        image_torch = Variable(images_tmp).cuda().unsqueeze(0)
                        res = ObjectDetector.detect_object_in_image_alreadytensor(
                            net,
                            pnp_solvers[model],
                            image_torch,
                            config_detect
                        )
                        # print("_runnetwork:res: ", res)
                        if res == []:
                            results_insg.append("test")
                        else:
                            results_insg.append(res)
                        i = i + 1
                    # print("_runnetwork:results_insg: ", results_insg)
                    # Find objects from network output
                    # print("_runnetwork:results: ", results)

                matrixHomogen = []
                for res in results_insg:
                    # print("for res in results_insg:")
                    if res == 'test':
                        matrixHomogen.append("test")
                        # print("asdfasfdsdf: ")
                    else:
                        # print("else")
                        for i_r, result in enumerate(res):  # enumerate(results):
                            # print("result: ", result)
                            if result["location"] is None:
                                continue
                            # print("_runnetwork:results: ", results)
                            object_name = result['name']  # welches objekt gefunden wurde
                            loc = result["location"]  # translation
                            ori = result["quaternion"]  # rotation

                            # Rx + T homogene Rigid body tramsformation
                            # quaternion zu Rotationsmatrix berechne + Transformation = Rx +T
                            quaternation = 0
                            quaternation = QtoR(ori)
                            # print("loadjson:quaternation: ", quaternation)
                            # print("loadjson:quaternation: ", type(quaternation))

                            tmp = np.insert(quaternation, 3, loc, axis=0)
                            # print("loadjson:insert: ", tmp)
                            # print("loadjson:insert:type: ", type(tmp))
                            # np.set_printoptions(precision=10)

                            resultPoseGU = np.insert(tmp, 3, [0, 0, 0, 1], axis=1)
                            matrixHomogen.append(resultPoseGU)
                            # print("loadjson:result: ", resultPoseGU)

                            # g_draw = ImageDraw.Draw(im)
                            # Draw the cube
                            # if None not in result['projected_points']:
                            #    points2d = []
                            #    for pair in result['projected_points']:
                            #        points2d.append(tuple(pair))
                            # projected_points.append(tuple(pair))

                # https://stats.stackexchange.com/questions/291820/what-is-the-definition-of-a-feature-map-aka-activation-map-in-a-convolutio
                # and feature maps??

                # pose_transform
                pose_transform = 0
                pose_transform = targets['pose_transform']
                # np.set_printoptions(precision=10)
                # pose_transform = Variable(targets['pose_transform'].cuda())
                # print("_runnetwork:pose_transform: ", pose_transform)
                # pose_transform = data['pose_transform']
                # print("_runnetwork:pose_transform: ", pose_transform)
                # print("_runnetwork:matrixHomogen: ", matrixHomogen)

                mean = 0
                i = 0  # Wie viele erkannt wurden np und fp
                image_size = len(pose_transform)  # How many Images wurden getestet

                # print("Test Metric:for tmp_pose_transform, tmp_matrixHomogen in zip(pose_transform, tmp_matrixHomogen):")
                for tmp_pose_transform, tmp_matrixHomogen in zip(pose_transform, matrixHomogen):
                    # print("Test Metric:pose_gt:pose_transform", tmp_pose_transform)
                    # print("Test Metric:pose_gu:tmp_matrixHomogen: ", tmp_matrixHomogen)
                    if tmp_matrixHomogen != 'test':
                        print("Test Metric:pose_gt:pose_transform", tmp_pose_transform)
                        print("Test Metric:pose_gu:tmp_matrixHomogen: ", tmp_matrixHomogen)
                        tmp = ADDErrorCuboid(tmp_pose_transform, tmp_matrixHomogen, cuboid)
                        print("Test Metric:ADDErrorCuboid:mean: ", tmp)
                        mean = mean + tmp
                        i = i + 1
                    # else:
                    # print("ADDErrorCuboid:mean: wrong")
                if mean != 0 and i > 1:
                    # print("ADDErrorCuboid:i: ", i)
                    mean = (mean / i)  # Durchschnitt berechnen
                    # print("ADDErrorCuboid:Durchschnitt:mean: ", mean)

                # the neural network outputs
                # belief maps to know where the cuboid object is located. Using that information and
                # the fact that we know the size and have access to the object size and the camera intrinsic
                # we find the 3d pose of the object using PnP.
                length_data = str(len(loader.dataset))
                length_data_count = len(length_data)
                test = batch_idx * len(data)
                asdf = str(test).zfill(length_data_count)
                print('Test Metric: {} [{}/{} ({:.0f}%)]\tMean: {:.15f}'.format(
                    epoch, asdf, len(loader.dataset),
                    100. * batch_idx / len(loader), mean))

                with open(opt.outf + '/test_metric.csv', 'a') as file:
                    # i = 0  # Wie viele erkannt wurden np und fp
                    # image_size = len(img_org)  # How many Images wurden getestet
                    s = '{}, {},{:.15f},{},{}\n'.format(
                        epoch, batch_idx, mean, i, image_size)
                    file.write(s)

                # End Py File here
                # quit()

                # --------------------------------------------------------------------------------------------------------------
                # delte files :
                # You don't even need to use rm in this case if you are afraid. Use find:
                #   find . -name "*.bak" -type f -delete
                # But use it with precaution. Run first:
                #   find . -name "*.bak" -type f

                # source devel/setup.bash
                # source /home/nils/catkin_ws/bin/activate



                # python test2.py --data /home/luedeke/Dataset_Luedeke_Neu/FerreroKuesschen/24000/ --datatest /home/luedeke/Dataset_Luedeke_Neu/FerreroKuesschen/Liv/ --object FerreroKuesschen --outf FerreroKuesschen
                # python test2.py --data /home/luedeke/Dataset_Luedeke_Neu/FerreroKuesschen/Livingroom/ --datatest /home/luedeke/Dataset_Luedeke_Neu/FerreroKuesschen/Liv/ --object FerreroKuesschen --outf FerreroKuesschen

                # python test2.py --data /home/luedeke/Dataset_Luedeke_Neu/FerreroKuesschen/Overfitting/ --datatest /home/luedeke/Dataset_Luedeke_Neu/FerreroKuesschen/Liv/ --object FerreroKuesschen --outf FerreroKuesschen

                # own pc
                # python test2.py --data /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/FerreroKuesschen/Overfitting/ --datatest /home/luedeke/Dataset_Luedeke_Neu/FerreroKuesschen/Liv/ --object FerreroKuesschen --outf FerreroKuesschen

                # python test2.py --data /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/FerreroKuesschen/Livingroom/Szene/ --datatest /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/FerreroKuesschen/Testdatensatz/ --object FerreroKuesschen --outf FerreroKuesschen

    # python test2.py --data /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/FerreroKuesschen/Livingroom/Szene/ --datatest /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/FerreroKuesschen/Testdatensatz/ --object FerreroKuesschen --outf FerreroKuesschen
                # Fat datensatz
                # result
                # python test2.py --data /media/nils/Ubuntu-TMP/CandyShop_neu/Livingroom/Szene1/Room_Capturer_ZED-Camera --datatest /media/nils/Ubuntu-TMP/Stereo_Dataset_Luedeke/single/CandyShop/Validation  --object CandyShop2 --outf CandyShop2

                # python test2.py --data /media/nils/Ubuntu-TMP/CandyShop_neu/CandyShop_Overfitting_test/ --datatest /media/nils/Ubuntu-TMP/Stereo_Dataset_Luedeke/single/CandyShop/Validation  --object CandyShop2 --outf CandyShop2

                # Overfitting Datensatz pc
                # python test2.py --data /media/nils/Ubuntu-TMP/fat/single/003_cracker_box_16k/kitch/ --datatest /media/nils/Ubuntu-TMP/fat/single/003_cracker_box_16k/kitch/ --object 003_cracker_box_16k --outf 003_cracker_box_16k

                # python test2.py --data /media/nils/Ubuntu-TMP/fat/single/003_cracker_box_16k/asdf/ --datatest /media/nils/Ubuntu-TMP/fat/single/003_cracker_box_16k/kitedemo_4/ --object 003_cracker_box_16k --outf 003_cracker_box_16k

                # Overfitting Datensatz pc
                # python test2.py --data /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/Dataset_Luedeke_Neu/CandyShop/CandyShop_Overfitting_100 --datatest /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/CandyShop/Livingroom/Szene1/Room_Capturer_ZED-Camera --object CandyShop2 --outf CandyShop2

                # python test2.py --data /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/CandyShop/CandyShop_Overfitting_1000 --datatest /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/CandyShop/Livingroom/Szene1/Room_Capturer_ZED-Camera --object CandyShop2 --outf CandyShop2
                # train
                # python test2.py --data /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/CandyShop/CandyShop_Overfitting_1000 --datatest /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/CandyShop/Livingroom/Szene1/Room_Capturer_ZED-Camera --object CandyShop2 --outf CandyShop2

                # python test2.py --data /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/FerreroKuesschen/Overfitting --datatest /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/FerreroKuesschen/Val/ --object FerreroKuesschen --outf FerreroKuesschen

                # python test2.py --data /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/FerreroKuesschen/Overfitting --datatest /media/nils/Ubuntu-TMP/Dataset_Luedeke_Neu/FerreroKuesschen/Overfitting --object FerreroKuesschen --outf FerreroKuesschen

                # cogsys rechner -----------------------------------------------------------------------------------------------
                # overfitting Datensatz
                # /home/luedeke/Stereo_Dataset_neu/Single/CandyShop/6000
                # python test2.py --data /home/luedeke/Stereo_Dataset_neu/Single/CandyShop/6000 --datatest /home/luedeke/Stereo_Dataset/Single/CandyShop/Validation --object CandyShop2 --outf CandyShop2

                # python test2.py --data /home/luedeke/Stereo_Dataset_neu/Single/CandyShop/CandyShop_Overfitting_3000 --datatest /home/luedeke/Stereo_Dataset/Single/CandyShop/Validation --object CandyShop2 --outf CandyShop2

                # python test2.py --data /home/luedeke/Stereo_Dataset_neu/Single/CandyShop/Livingroom --datatest /home/luedeke/Stereo_Dataset_neu/Single/CandyShop/Validation --object CandyShop2 --outf CandyShop2

                # grossser Datensatz
                # python test2.py --data /home/luedeke/Stereo_Dataset/Single/CandyShop/RoomAndBerlin --datatest /home/luedeke/Stereo_Dataset/Single/CandyShop/Validation --object CandyShop2 --outf CandyShop2

                # kleiner Datensatz
                # python test2.py --data /home/luedeke/Stereo_Dataset/Single/CandyShop/RoomAndBerlin/Room --datatest /home/luedeke/Stereo_Dataset/Single/CandyShop/Validation   --object CandyShop2 --outf CandyShop2

                # python test2.py --data /media/nils/Ubuntu-TMP/Stereo_Dataset_Luedeke/ --object FerreroKuesschen --outf FerreroKuesschen
                # rsync -avz /media/nils/Ubuntu-TMP/Stereo_Dataset_Luedeke/single/ luedeke@dlsys-MACHINE:/home/luedeke/Stereo_Dataset/Single

                # rsync -avz /media/nils/Ubuntu-TMP/CandyShop_neu/CandyShop_Overfitting_3000/ luedeke@dlsys-MACHINE:/home/luedeke/Stereo_Dataset_neu/Single/CandyShop/CandyShop_Overfitting_3000/

                # Compute the Metric -------------------------------------------------------------------------------------------
        # break
        if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
            torch.save(net.state_dict(), '{}/net_{}.pth'.format(opt.outf, opt.namefile))
            break


        # imshow(torchvision.utils.make_grid(data))
        # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))



for epoch in range(1, opt.epochs + 1):
    if not trainingdata is None:
        _runnetwork(epoch, trainingdata)

    if not opt.datatest == "":
        _runnetwork(epoch, testingdata, train=False)
        if opt.data == "":
            break  # lets get out of this if we are only testing
    try:
        torch.save(net.state_dict(), '{}/net_{}_{}.pth'.format(opt.outf, opt.namefile, epoch))
    except:
        pass

    if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
        break

#Finished Training/Fine-tuning -----------------------------------------------------------------------------------------
end = datetime.datetime.now().time()
print("end: ", end)
with open (opt.outf+'/header.txt','a') as file:
    file.write("\nend: " + str(end)+"\n")

#Finished then print table loss-----------------------------------------------------------------------------------------
#Created by: Luedeke
#Plot Graph from CSV file from DOPE
# Epoch and Loss or something lese...

files = ['loss_test','loss_train']
for file in files:
    x = []
    y = []

    #with open(opt.outf+'/loss_test.csv', 'r') as csvfile:
    with open(opt.outf + '/' + file + '.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots)
        for row in plots:
        #print(row[0])
            x.append(int(row[0]))       #epoche
            y.append(float(row[2]))     #loss
            #zy.append(float(row[2]))   #batchsize

        epoche = 1
        tmp = 0
        tmp_index = 0

        tmpx = []
        tmpy = []

        #print 0.000099009012047 #9.9009012047e-05
        #print 0.000153484608745

        #print len(x)
        for i in xrange(len(x)):
            #if(0.0001 < y[i]):
            #    print y[i]
            #print y[i], 'Epoche: ', x[i]#, ' Index: ', i
            if(x[i] == epoche):
                tmp += y[i]     #durchschnitt
                tmp_index += 1
                #print 'tmp: ', tmp, ' index: ', tmp_index
            else:
                #print 'Epoche: ', epoche
                #durchschnitt berechnen
                tmpx.append(epoche)
                tmpy.append(tmp / tmp_index)
                #print 'tmp: ', tmp, ' index: ', tmp_index
                #print 'Loss: ', (tmp/tmp_index)
                #print 'Laenge: ', len(tmpx)
                #print(tmpx[epoche-1])
                #print(tmpy[epoche-1])

                tmp_index = 1
                tmp = 0
                tmp += y[i]
                epoche = x[i]
                #print 'tmp: ', tmp, ' index: ', tmp_index
            #if (i == 1650):
            #    break

        #for i in xrange(len(tmpy)):
        #   print(tmpy[i])

    #plt.plot(tmpx,tmpy, label='DOPE: Fine-Tuning Layer:' + str(deleted_layers))
    #plt.xlabel('Epoche')
    #plt.ylabel('Loss')
    #plt.title('DOPE')
    #plt.legend()
    #plt.show()
    #plt.savefig(file + '.png', dpi=300)

#Finished then print table loss-----------------------------------------------------------------------------------------


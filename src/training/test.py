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

        vgg_full = models.vgg19(pretrained=False).features
        self.vgg = nn.Sequential()
        for i_layer in range(24):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        # Add some layers
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


def loadjson(path, objectsofinterest, img):
    """
    Loads the data from a json file.
    If there are no objects of interest, then load all the objects.
    """
    with open(path) as data_file:
        data = json.load(data_file)
    # print (path)
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

    for i_line in range(len(data['objects'])):
        info = data['objects'][i_line]
        if not objectsofinterest is None and \
                not objectsofinterest in info['class'].lower():
            continue

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

        # load translations
        location = info['location']
        translations.append([location[0], location[1], location[2]])

        # quaternion
        rot = info["quaternion_xyzw"]
        rotations.append(rot)

    return {
        "pointsBelief": pointsBelief,
        "rotations": rotations,
        "translations": translations,
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
        for imgpath in glob.glob(path + "/*.png"):
            if exists(imgpath) and exists(imgpath.replace('png', "json")):
                imgs.append((imgpath, imgpath.replace(path, "").replace("/", ""),
                             imgpath.replace('png', "json")))

    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [os.path.join(path, o) for o in os.listdir(path)
                   if os.path.isdir(os.path.join(path, o))]
        if len(folders) > 0:
            for path_entry in folders:
                explore(path_entry)
        else:
            add_json_files(path)

    explore(root)

    return imgs


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

        def load_data(path):
            '''Recursively load the data.  This is useful to load all of the FAT dataset.'''
            imgs = loadimages(path)

            # Check all the folders in path
            for name in os.listdir(str(path)):
                imgs += loadimages(path + "/" + name)
            return imgs

        self.imgs = load_data(root)

        # Shuffle the data, this is useful when we want to use a subset.
        np.random.shuffle(self.imgs)

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
        img_size = (400, 400)

        loader = loadjson

        data = loader(txt, self.objectsofinterest, img)

        pointsBelief = data['pointsBelief']
        objects_centroid = data['centroids']
        points_all = data['points']
        points_keypoints = data['keypoints_2d']
        translations = torch.from_numpy(np.array(
            data['translations'])).float()
        rotations = torch.from_numpy(np.array(
            data['rotations'])).float()

        if len(points_all) == 0:
            points_all = torch.zeros(1)

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
            Reprojection of points when rotating the image
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
        dx = round(np.random.normal(0, 2) * float(self.random_translation[0]))
        dy = round(np.random.normal(0, 2) * float(self.random_translation[1]))
        angle = round(np.random.normal(0, 1) * float(self.random_rotation))

        tm = np.float32([[1, 0, dx], [0, 1, dy]])
        rm = cv2.getRotationMatrix2D(
            (img.size[0] / 2, img.size[1] / 2), angle, 1)

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
        # This is used when we do saving --- helpful for debugging
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

        return {
            'img': img,
            "affinities": affinities,
            'beliefs': beliefs,
        }




#-----------------------------------------------------------------------------------------------------------------------
# Finetuning -----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

##################################################
#Load Model
#cracker_60_Kopie.pth model
# Initialize model
##################################################

model = DopeNetwork().cuda()
model = torch.nn.DataParallel(net,device_ids=opt.gpuids).cuda()
model.load_state_dict(torch.load('cracker_60_Kopie.pth'))
model.eval()


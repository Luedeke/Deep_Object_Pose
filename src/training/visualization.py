#!/usr/bin/env python
# Hi guys, here is my code for visualizing the feature maps. The detector is provided by DOPE. Hope it helps.
#Created by Abdul-Mukit from: https://github.com/NVlabs/Deep_Object_Pose/issues/4#issuecomment-475374866
#Edited by: Luedeke

from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import torch


# Import DOPE code
import sys
import rospkg
rospack = rospkg.RosPack()
g_path2package = rospack.get_path('dope')
sys.path.append("{}/src/inference".format(g_path2package))
from detector import *



## Settings
name = 'CandyShop2'
#net_path = '/media/nils/Seagate\ Expansion\ Drive/Thesis/Trainiert/train_CandyShop2_01_05/net_epoch_60.pth'
#net_path = '/media/nils/Seagate Expansion Drive/Thesis/Trainiert/train_CandyShop2_02_05/net_epoch_60.pth'
#net_path = '/home/nils/catkin_ws/src/dope/weights/soup_60.pth' #soup_60  #cracker_60

net_path = '/media/nils/Seagate Expansion Drive/Thesis/Trainiert/'
file     = 'train_CandyShop2_22_05_19-full-training/'
#train_CandyShop2_12_05_overfitting_3000_
#train_CandyShop2_3000_09_05_overfitting
#train_CandyShop2_03_05
#train_CandyShop2_02_05
epoche   = 'net_epoch_60.pth'
net_path = net_path + file + epoche


#data/net/
# net_path = '/net_epoch_42'
gpu_id = 0
img_path = 'CandyShop2.png'
#img_path = '/home/nils/catkin_ws/src/dope/weights/dope_objects.png' #dope_objects #soup
# img_path = '/CandyShop2.jpg'



# Function for visualizing feature maps
def viz_layer(layer, n_filters=9):
    fig = plt.figure(figsize=(20, 20))
    row = 1
    for i in range(n_filters):
        ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow(np.squeeze(layer[i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i + 1))

# load color image
in_img = cv2.imread(img_path)

if os.path.isfile(img_path):
    print ("Loading is sucessfull.")
else:
    print ("The file " + img_path + " does not exist.")

#in_img = cv2.resize(in_img, (640, 480))
in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB) #COLOR_BGR2GRAY         COLOR_BGR2RGB


# plot image
plt.imshow(in_img)


model = ModelData(name, net_path, gpu_id)
model.load_net_model()
net_model = model.net

# Run network inference
image_tensor = transform(in_img)
image_torch = Variable(image_tensor).cuda().unsqueeze(0)
out, seg = net_model(image_torch)
vertex2 = out[-1][0].cpu()
aff = seg[-1][0].cpu()

# View the vertex and affinities
viz_layer(vertex2)
viz_layer(aff, n_filters=16)

plt.show()


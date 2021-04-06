from __future__ import print_function, division
import os
import torch
import random as rn
import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms

import torchvision.transforms.functional as F

import pandas as pd
from PIL import Image

import utils as utils

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import random

class SimulationDataset(Dataset):
    """Dataset wrapping input and target tensors for the driving simulation dataset.

    Arguments:
        set (String):  Dataset - train, test
        path (String): Path to the csv file with the image paths and the target values
    """

    def __init__(self, set, csv_path='driving_log.csv', transforms=None):

        self.transforms = transforms

        self.data = pd.read_csv(csv_path, header=None)

        # First column contains the middle image paths
        # Fourth column contains the steering angle
        start = int(random.random() * 1/5 * len(self.data))
        end = int(start + 4/5 * len(self.data))

        WAYPOINT_COUNT = 30
        WAYPOINT_OFFSET = 7
        WAYPOINT_END = WAYPOINT_OFFSET + WAYPOINT_COUNT 


        if (set == "test"):
            start = end 
            end = len(self.data)

        # Just know that these are the paths to the image we will open with PIL 
        self.image_paths = np.array(self.data.iloc[start:end, 0:3])

        # these are 30 comma separated values that are the x, y, z coordinates
        # of each waypoint in our CSV file (written in from Unity)
        self.waypoints = np.array(self.data.iloc[start:end,WAYPOINT_OFFSET:WAYPOINT_END])

        print(self.waypoints);

        

    def __getitem__(self, index):

         # Get image name from the pandas df
        image_paths = self.image_paths[index]
        # Open image
        image = [Image.open(image_paths[i]) for i in range(3)]

        # Getting the 30 waypoint associated with the image
        waypoints = self.waypoints[index]     

        sample = {'image': image, 'target': waypoints}

        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        if self.transforms is not None:
            sample = self.transforms(sample)

        # plt.imshow(F.to_pil_image(sample['image']))
        # plt.title(str(sample['target']))
        # plt.show()
        
        return sample['image'], sample['target']

    def __len__(self):
        return len(self.image_paths)


if  __name__ =='__main__':

    input_shape = (utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH)
    dataset = SimulationDataset("train", transforms=transforms.Compose([                 
                utils.RandomCoose(['center']),          
                utils.Preprocess(input_shape),
                utils.RandomHorizontalFlip(),
                utils.ToTensor(),
                utils.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))
    print(dataset.__getitem__(0)[1])
    print(dataset.__len__())
    print(dataset.__getitem__(0)[0].size())

#    for c in range(3):
#        for i in range(dataset.__len__()):
#            print(dataset.__getitem__(i)[c].mean())
#            print(dataset.__getitem__(i)[c].std())
    # print(dataset.__getitem__(0))
    # print(len(dataset.__get_annotations__()))

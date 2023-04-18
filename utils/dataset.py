import os
import json
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torchvision.transforms as transforms
import time

# Personalized dataset class

class Dataset(data.Dataset):
    def __init__(self, dir_path, transform=None, dataset='tr'):
        # Initialization
        self.tiles_path = dir_path
        self.transform = transform
        self.dataset = dataset
        
        # Choose dataset to use
        
        if dataset=='comb_tr':
            tiles_num_dict_path = './data/num_tiles_dict_comb_tr.json'
            
        elif dataset=='comb_val':
            tiles_num_dict_path = './data/num_tiles_dict_comb_val.json'
            
        elif dataset=='comb_ts':
            tiles_num_dict_path = './data/num_tiles_dict_comb_ts.json'
        
        elif dataset=='ts':
            tiles_num_dict_path = './data/num_tiles_dict_ts.json'
            
        elif dataset=='ext':
            tiles_num_dict_path = './data/num_tiles_dict_ext.json'
        
        elif dataset=='ext_rand':
                tiles_num_dict_path = './data/num_tiles_dict_ext_rand.json'
            
        else:
            tiles_num_dict_path = './data/num_tiles_dict_tr.json'
            
        with open(tiles_num_dict_path) as json_file:
            self.num_tiles_dict = json.load(json_file)

    def __len__(self):
        #Denotes the total number of samples
        num_samples = 0
        for n in self.num_tiles_dict.values():
            num_samples += n
        return num_samples


    def __getitem__(self, index):
        #start_time = time.time()
        
        # Generates one sample of data
        # Select sample by looking in which slide folder the tile is
        curr_tiles_sum = 0
        
        for slide_name in self.num_tiles_dict:
            if index < curr_tiles_sum + self.num_tiles_dict[slide_name]:
                selected_slide_name = slide_name
                break
            curr_tiles_sum += self.num_tiles_dict[slide_name]
        
        tile_idx = index - curr_tiles_sum
        
       
        #start_time_idx = time.time()
 
        tile_path = os.path.join(self.tiles_path, selected_slide_name, str(tile_idx) + '.jpg')
        
        #end_time_idx = time.time()
        #print('time idx: ', "{:.8f}".format(end_time_idx - start_time_idx))
        X = Image.open(tile_path)

        if self.transform:
            X = self.transform(X)     # transform

        #end_time = time.time()
        #print('time get: ', "{:.8f}".format(end_time - start_time))
        
        return X
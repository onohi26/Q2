from __future__ import print_function, division
import os
import torch
import cv2
from torch.autograd import Variable
from skimage import io
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf
from geotnf.transformation import homography_mat_from_4_pts

class SynthDataset(Dataset):
    """
    
    Synthetically transformed pairs dataset for training with strong supervision
    
    Args:
            csv_file (string): Path to the csv file with image names and transformations.
            training_image_path (string): Directory with all the images.
            transform (callable): Transformation for post-processing the training pair (eg. image normalization)
            
    Returns:
            Dict: {'image': full dataset image, 'theta': desired transformation}
            
    """

    def __init__(self,
                 dataset_csv_path, 
                 dataset_csv_file, 
                 dataset_image_path, 
                 output_size=(256,256), 
                 geometric_model='affine', 
                 dataset_size=0, 
                 transform=None, 
                 random_sample=False, 
                 random_t=0.5, 
                 random_s=0.5, 
                 random_alpha=1/6, 
                 random_t_tps=0.4, 
                 four_point_hom=True):
    
        self.out_h, self.out_w = output_size
        # read csv file
        self.train_data = pd.read_csv(os.path.join(dataset_csv_path,dataset_csv_file))
        self.random_sample = random_sample
        self.random_t = random_t
        self.random_t_tps = random_t_tps
        self.random_alpha = random_alpha
        self.random_s = random_s
        self.four_point_hom = four_point_hom
        self.dataset_size = dataset_size
        if dataset_size!=0:
            dataset_size = min((dataset_size,len(self.train_data)))
            self.train_data = self.train_data.iloc[0:dataset_size,:]
        self.img_names = self.train_data.iloc[:,0]
        if self.random_sample==False:
            self.theta_array = self.train_data.iloc[:, 1:].values().astype('float')
        # copy arguments
        self.dataset_image_path = dataset_image_path
        self.transform = transform
        self.geometric_model = geometric_model
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        if self.random_sample and self.dataset_size==1:
            np.random.seed(1) # for debugging purposes
        # read image
        img_name = os.path.join(self.dataset_image_path, self.img_names[idx])
        #image = io.imread(img_name)
        with open(img_name, 'rb') as f:
            arr = np.fromfile(f, dtype=np.float64, count=256*256)
            image = arr.reshape((256, 256))
            image = (image-image.min())/(image.max() - image.min())*255
            image = cv2.cvtColor(np.uint8(image),cv2.COLOR_GRAY2BGR)

        # read theta
        if self.random_sample==False:
            theta = self.theta_array[idx, :]

            if self.geometric_model=='affine':
                # reshape theta to 2x3 matrix [A|t] where 
                # first row corresponds to X and second to Y
    #            theta = theta[[0,1,4,2,3,5]].reshape(2,3)
                theta = theta[[3,2,5,1,0,4]] #.reshape(2,3)
            if self.geometric_model=='tps':
                theta = np.expand_dims(np.expand_dims(theta,1),2)
            if self.geometric_model=='afftps':
                theta[[0,1,2,3,4,5]] = theta[[3,2,5,1,0,4]]
        else:
            if self.geometric_model=='affine' or self.geometric_model=='afftps':
                rot_angle = (np.random.rand(1)-0.5)*2*np.pi/12; # between -np.pi/12 and np.pi/12
                sh_angle = (np.random.rand(1)-0.5)*2*np.pi/6; # between -np.pi/6 and np.pi/6
                lambda_1 = 1+(2*np.random.rand(1)-1)*0.25; # between 0.75 and 1.25
                lambda_2 = 1+(2*np.random.rand(1)-1)*0.25; # between 0.75 and 1.25
                tx=(2*np.random.rand(1)-1)*0.25;  # between -0.25 and 0.25
                ty=(2*np.random.rand(1)-1)*0.25;

                R_sh = np.array([[np.cos(sh_angle[0]),-np.sin(sh_angle[0])],
                                 [np.sin(sh_angle[0]),np.cos(sh_angle[0])]])
                R_alpha = np.array([[np.cos(rot_angle[0]),-np.sin(rot_angle[0])],
                                    [np.sin(rot_angle[0]),np.cos(rot_angle[0])]])

                D=np.diag([lambda_1[0],lambda_2[0]])

                A = R_alpha @ R_sh.transpose() @ D @ R_sh

                theta_aff = np.array([A[0,0],A[0,1],tx,A[1,0],A[1,1],ty])
            if self.geometric_model=='hom':
                theta_hom = np.array([-1, -1, 1, 1, -1, 1, -1, 1])
                theta_hom = theta_hom+(np.random.rand(8)-0.5)*2*self.random_t_tps
            if self.geometric_model=='tps' or self.geometric_model=='afftps':
                theta_tps = np.array([-1 , -1 , -1 , 0 , 0 , 0 , 1 , 1 , 1 , -1 , 0 , 1 , -1 , 0 , 1 , -1 , 0 , 1])
                theta_tps = theta_tps+(np.random.rand(18)-0.5)*2*self.random_t_tps
            if self.geometric_model=='affine':
                theta=theta_aff
            elif self.geometric_model=='hom':
                theta=theta_hom
            elif self.geometric_model=='tps':
                theta=theta_tps
            elif self.geometric_model=='afftps':
                theta=np.concatenate((theta_aff,theta_tps))
            
        # make arrays float tensor for subsequent processing
        image = torch.Tensor(image.astype(np.float32))
        theta = torch.Tensor(theta.astype(np.float32))

        if self.geometric_model=='hom' and self.four_point_hom==False:
            theta = homography_mat_from_4_pts(Variable(theta.unsqueeze(0))).squeeze(0).data
            # theta = torch.div(theta[:8],theta[8])
        
        # permute order of image to CHW
        image = image.transpose(1,2).transpose(0,1)
                
        # Resize image using bilinear sampling with identity affine tnf
        if image.size()[0]!=self.out_h or image.size()[1]!=self.out_w:
            image = self.affineTnf(Variable(image.unsqueeze(0),requires_grad=False)).data.squeeze(0)
                
        sample = {'image': image, 'theta': theta}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

class IXIDataset(Dataset):
    
    """
    
    Proposal Flow image pair dataset
    

    Args:
        csv_file (string): Path to the csv file with image names and transformations.
        dataset_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)
        
    """

    def __init__(self, json_path,dataset_path,output_size=(256,256), transform=None):

        self.out_h, self.out_w = output_size
        json_file = open(json_path)
        json_dict = json.load(json_file)
        self.img_A_names = []
        self.img_B_names = []
        self.point_coords = []
        for dic in json_dict:
            self.img_A_names.append(os.path.join('images_source',dic["filename"]))
            self.img_B_names.append(os.path.join('images_target',dic["filename"]))
            keypoints = dic['keypoints']
            point_coord = []
            for keypoint in keypoints:
                point_coord.append(keypoint['pixel'][0])
            for keypoint in keypoints:
                point_coord.append(keypoint['pixel'][1])
            self.point_coords.append(point_coord)
        self.point_coords = np.asarray(self.point_coords)
        self.dataset_path = dataset_path      
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
              
    def __len__(self):
        return len(self.img_A_names)

    def __getitem__(self, idx):
        # get pre-processed images
        image_A,label_A,im_size_A, basename_A = self.get_image(self.img_A_names,idx)
        image_B,label_B,im_size_B, basename_B = self.get_image(self.img_B_names,idx)

        point_coords = self.get_points(self.point_coords,idx)
                
        sample = {'source_image': image_A, 'target_image': image_B, 'source_label':label_A,'target_label':label_B,'source_im_size': im_size_A, 'target_im_size': im_size_B,
        'source_name':basename_A, 'target_name':basename_B, 'source_points': point_coords,}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self,img_name_list,idx):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        with open(img_name, 'rb') as f:
            arr = np.fromfile(f, dtype=np.float64, count=256*256)
            image = arr.reshape((256, 256))
            image = (image-image.min())/(image.max() - image.min())*255
        
        label = self.get_label(np.uint8(image))
        image = cv2.cvtColor(np.uint8(image),cv2.COLOR_GRAY2BGR)
                
        # get image size
        im_size = np.asarray(image.shape)
        
        # convert to torch Variable
        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image,requires_grad=False)
        
        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)
        
        im_size = torch.Tensor(im_size.astype(np.float32))
        
        return (image, label, im_size, os.path.splitext(os.path.basename(img_name_list[idx]))[0])
    
    def get_points(self,point_coords_list,idx):
        point_coords = point_coords_list[idx,:].reshape(2,12)

#        # swap X,Y coords, as the the row,col order (Y,X) is used for computations
#        point_coords = point_coords[[1,0],:]

        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords
    
    def get_label(self,image):
        _, binary_img = cv2.threshold(image, 30, 20, cv2.THRESH_BINARY)
        close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        _,filled_img,_,_ = cv2.floodFill(close_img,None,(0,0),255)
        _,binary_filled_img = cv2.threshold(filled_img, 0, 255, cv2.THRESH_OTSU)
        inverted_img = cv2.bitwise_not(binary_filled_img)
        image = cv2.cvtColor(np.uint8(inverted_img),cv2.COLOR_GRAY2BGR)
        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image,requires_grad=False)
        image = self.affineTnf(image_var).data.squeeze(0)

        return image
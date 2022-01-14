from __future__ import print_function, division
import os
import argparse
from torch.utils.data import DataLoader
from model.cnn_geometric_model import CNNGeometric
from data.affine_dataset import IXIDataset
from image.normalization_affine import NormalizeImageDict, normalize_image
from util.torch_util import BatchTensorToVars
from geotnf.transformation import GeometricTnf
from geotnf.point_tnf import *
import matplotlib.pyplot as plt
from skimage import io,img_as_ubyte
from collections import OrderedDict
from geotnf.transformation import homography_mat_from_4_pts
from util.torch_util import str_to_bool
from model.loss import NCC
import torch.optim as optim
import torch
import torch.nn as nn
import json

"""

Script to demonstrate evaluation on a trained model as presented in the CNNGeometric CVPR'17 paper
on the ProposalFlow dataset

"""

def tensor2numpy(tensor):
    return tensor.detach().cpu().numpy()

def transform_points(points, theta):
    '''入力された点をthetaに従って移動させる関数'''
    theta = theta.reshape(2,3)
    mat = np.eye(3)
    mat[:2] = theta
    mat = np.linalg.inv(mat)
    rot_mat = mat[:2, :2]
    trans = mat[:2, 2] * 128

    trans_points = (rot_mat @ (points - 128).T).T[:, :2] + 128
    trans_points[:, 0] = trans_points[:, 0] + trans[0]
    trans_points[:, 1] = trans_points[:, 1] + trans[1]
    return np.round(trans_points)


print('CNNGeometric demo script')

# Argument parsing
parser = argparse.ArgumentParser(description='CNNGeometric PyTorch implementation')
# Paths
parser.add_argument('--model-aff', type=str,
                    default='./trained_models/checkpoint_adam/best_checkpoint_adam_affine_mse_loss_vgg.pth.tar',
                    help='Trained affine model filename')
parser.add_argument('--model-hom', type=str,
                    default='',
                    help='Trained homography model filename')
parser.add_argument('--feature_extraction_cnn', type=str, default='vgg',
                    help='Feature extraction architecture: vgg/resnet101')
parser.add_argument('--json-path', type=str, default='keypoints_source.json', help='Path to target images')
parser.add_argument('--dataset_path', type=str, default='./datasets', help='Path to PF dataset')
parser.add_argument('--result_path', type=str, default='./datasets/result_affine', help='Path to result')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--num-epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--loss-function', type=str, default='mse', help='ncc/mse')


args = parser.parse_args()

json_list = []

use_cuda = torch.cuda.is_available()

do_aff = not args.model_aff == ''
do_hom = not args.model_hom == ''

# Create model
print('Creating CNN model...')
if do_aff:
    model = CNNGeometric(use_cuda=use_cuda,
                             feature_extraction_cnn=args.feature_extraction_cnn)

if do_hom:
    model = CNNGeometric(use_cuda=use_cuda, output_dim=8,
                             feature_extraction_cnn=args.feature_extraction_cnn)

# Load trained weights
print('Loading trained model weights...')
if do_aff:
    checkpoint = torch.load(args.model_aff, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict(
        [(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])

if do_hom:
    checkpoint = torch.load(args.model_hom, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict(
        [(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
    

# Dataset and dataloader
dataset = IXIDataset(json_path=os.path.join(args.dataset_path, args.json_path),
                    dataset_path=args.dataset_path,
                    transform=NormalizeImageDict(['source_image', 'target_image']))
dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
batchTensorToVars = BatchTensorToVars(use_cuda=use_cuda)

# Instantiate point transformer
pt = PointTnf(use_cuda=use_cuda)

# Instatiate image transformers
affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)
homTnf = GeometricTnf(geometric_model='hom', use_cuda=use_cuda)

for i, batch in enumerate(dataloader):
    # get random batch of size 1
    print('[number:',i,']')
    batch = batchTensorToVars(batch)

    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']
    source_im_name = batch['source_name'][0]
    target_im_name = batch['target_name'][0]
    source_points = batch['source_points']

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


    # Evaluate models
    if do_aff:
        theta_aff = model(batch)
        theta_aff = theta_aff.detach()
        theta_mat = theta_aff.view(-1,2,3)
        source_image = normalize_image(batch['source_image'], forward=False)
        warped_image_aff = affTnf(source_image, theta_mat)
        theta = tensor2numpy(theta_aff)
        source_points = tensor2numpy(source_points)
        warped_points_aff = transform_points(source_points,theta)

    elif do_hom:
        theta_hom = model(batch)
        theta_hom=theta_hom.detach()
        source_image = normalize_image(batch['source_image'], forward=False)
        warped_image_aff = homTnf(source_image, theta_hom)


    #Optimization for each image pair

    if args.loss_function == 'mse':
        criterion=nn.MSELoss(reduction='sum')
    if args.loss_function =='ncc':
        criterion=NCC()
    optimizer = optim.Adam(model.FeatureRegression.parameters(), lr=args.lr)
  
    tnf_batch = {'source_image': normalize_image(warped_image_aff), 'target_image': batch['target_image']}
    target_image = normalize_image(batch['target_image'],forward=False)

    model.train()
    for epoch in range(1,args.num_epochs+1):
        optimizer.zero_grad()
        theta = model(tnf_batch)
        if do_aff:
            theta_mat = theta.view(-1, 2, 3)
            warped_image = affTnf(warped_image_aff,theta_mat)
            loss = criterion(warped_image, target_image)
        elif do_hom:
            warped_image = homTnf(warped_image_aff, theta)
            loss = criterion(warped_image, target_image)
        loss.backward()
        optimizer.step()
    model.eval()

    # Evaluate models
    if do_aff:
        theta_aff = model(tnf_batch)
        theta_mat = theta_aff.view(-1,2,3)
        warped_image_aff = affTnf(warped_image_aff, theta_mat)
        theta = tensor2numpy(theta_aff)
        warped_points_aff = transform_points(warped_points_aff,theta)
    elif do_hom:
        theta_hom = model(tnf_batch)
        warped_image_aff = homTnf(tnf_batch['source_image'], theta_hom)


    # Un-normalize images and convert to numpy
    source_image = source_image.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    target_image = normalize_image(batch['target_image'], forward=False)
    target_image = target_image.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    warped_image_aff = warped_image_aff.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()



    print('Writing results to:')
    fn_aff = os.path.join(args.result_path, source_im_name+'.bmp')
    print(fn_aff)
    io.imsave(fn_aff, img_as_ubyte(warped_image_aff))
    warped_points_list = warped_points_aff.tolist()[0]
    dic = {
            "filename": source_im_name +'.raw',
            "image_size": [
            256,
            256
            ],
            "keypoints": [{"id": id,"pixel":[warped_points_list[0][id],warped_points_list[1][id]]} for id in range(12)]
            }
    json_list.append(dic)

json_file = open('./datasets/affine.json','w')
json.dump(json_list,json_file,indent=0)
    
    
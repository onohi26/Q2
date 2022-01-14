from __future__ import print_function, division
import os
import argparse
import torch
from torch.utils.data import DataLoader
from data.vxm_dataset import IXIDataset
from image.normalization_vxm import NormalizeImageDict, normalize_image
from util.torch_util import BatchTensorToVars
from geotnf.transformation import GeometricTnf
import matplotlib.pyplot as plt
from skimage import io
from collections import OrderedDict
import json
import voxelmorph as vxm 
import torch.nn as nn
import torch.optim as optim

print('CNNGeometric demo script')

# Argument parsing
parser = argparse.ArgumentParser()
# Paths
parser.add_argument('--model', type=str,
                    default='./trained_models/checkpoint_adam/best_checkpoint_adam_mse_loss_vxm.pth.tar',
                    help='Trained model filename')
parser.add_argument('--dataset-path', type=str, default='./datasets', help='Path to IXI dataset')
parser.add_argument('--source-path', type=str, default='result_affine', help='Path to source images')
parser.add_argument('--source-suffix', type=str, default='.bmp', help='Path to source images')
parser.add_argument('--target-path', type=str, default='images_target', help='Path to target images')
parser.add_argument('--json-path', type=str, default='affine.json', help='Path to target images')
parser.add_argument('--result_path', type=str, default='./datasets/result_vxm', help='Path of the result')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--num-epochs', type=int, default=100, help='number of training epochs')

args = parser.parse_args()

json_list = []

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')

enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

# Create model
print('Creating CNN model...')
model = vxm.networks.VxmDense(inshape = [256,256],
                         nb_unet_features=[enc_nf,dec_nf],
                         int_steps = args.int_steps,
                         int_downsize=args.int_downsize)

# Load trained weights
print('Loading trained model weights...')
checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)


# Dataset and dataloader
dataset = IXIDataset(json_path=os.path.join(args.dataset_path, args.json_path),
                    dataset_path=args.dataset_path,
                    source_path=args.source_path,target_path=args.target_path,source_suffix=args.source_suffix,
                    transform=NormalizeImageDict(['source_image', 'target_image']))
dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=4)
batchTensorToVars = BatchTensorToVars(use_cuda=use_cuda)

pt_transformer = vxm.layers.PointSpatialTransformer([256,256])
pt_transformer.to(device)
fullsize = vxm.layers.ResizeTransform(1 / args.int_downsize, 2)

criterion=nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=args.lr)

for i, batch in enumerate(dataloader):
    # get random batch of size 1
    print('[number:',i,']')
    model.load_state_dict(checkpoint['state_dict'])

    batch = batchTensorToVars(batch)

    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']

    source_im_name = batch['source_name'][0]
    target_im_name = batch['target_name'][0]

    source_points = batch['source_points']

    model.to(device)
    model.train()

    #Instance specific optimization
    for epoch in range(1,args.num_epochs+1):
        optimizer.zero_grad()
        moved,warp = model(batch)
        loss = criterion(batch['target_image'], moved)
        loss.backward()
        optimizer.step()

    # Evaluate models
    model.eval()    
    moved,warp = model(batch)
    flow = fullsize(warp)
    warped_points = pt_transformer(source_points,flow)
    # Un-normalize images and convert to numpy
    source_image = normalize_image(batch['source_image'], forward=False)
    source_image = source_image.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    target_image = normalize_image(batch['target_image'], forward=False)
    target_image = target_image.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    warped_image = moved*255
    warped_image = warped_image.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    fn_aff = os.path.join(args.result_path, source_im_name+'.bmp')
    print(fn_aff)
    io.imsave(fn_aff, warped_image)
    dic = {
            "filename": source_im_name +'.raw',
            "image_size": [
            256,
            256
            ],
            "keypoints": [{"id": id,"pixel":[int(warped_points[0][id]),int(warped_points[1][id])]} for id in range(12)]
            }
    json_list.append(dic)


json_file = open('vxm.json','w')
json.dump(json_list,json_file,indent=0) 
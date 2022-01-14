from __future__ import print_function, division
import argparse
import os
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from data.vxm_dataset import SynthDataset

from geotnf.transformation import SynthPairTnf

from image.normalization_vxm import NormalizeImageDict

from util.train_test_fn_vxm import train, validate_model
from util.torch_util import save_checkpoint, str_to_bool

from options.options_vxm import ArgumentParser

import voxelmorph as vxm

os.environ['VXM_BACKEND'] = 'pytorch'

"""

Script to evaluate a trained model as presented in the CNNGeometric TPAMI paper
on the PF/PF-pascal/Caltech-101 and TSS datasets

"""

def main():

    args,arg_groups = ArgumentParser(mode='train').parse()
    print(args)

    bidir = args.bidir

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    # Seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    if args.training_dataset == 'ixi':

        if args.dataset_image_path == '':
            args.dataset_image_path = 'datasets/'

        args.dataset_csv_path = 'training_data/ixi-random'    


    # CNN model and loss
    print('Creating CNN model...')

    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    model = vxm.networks.VxmDense(inshape = [256,256],
                         nb_unet_features=[enc_nf,dec_nf],
                         bidir=bidir,
                         int_steps = args.int_steps,
                         int_downsize=args.int_downsize)

    model.to(device)
    model.train()

    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)
  

    # Initialize Dataset objects
    dataset = SynthDataset(geometric_model=args.geometric_model,
               dataset_csv_path=args.dataset_csv_path,
               dataset_csv_file='train.csv',
			   dataset_image_path=args.dataset_image_path,
			   transform=NormalizeImageDict(['image']),
			   random_sample=args.random_sample)

    dataset_val = SynthDataset(geometric_model=args.geometric_model,
                   dataset_csv_path=args.dataset_csv_path,
                   dataset_csv_file='val.csv',
			       dataset_image_path=args.dataset_image_path,
			       transform=NormalizeImageDict(['image']),
			       random_sample=args.random_sample)

    # Set Tnf pair generation func
    pair_generation_tnf = SynthPairTnf(geometric_model=args.geometric_model,
				       use_cuda=use_cuda)

    # Initialize DataLoaders
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4)

    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size,
                                shuffle=True, num_workers=4)

    # Optimizer and eventual scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.lr_max_iter,
                                                               eta_min=1e-6)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    else:
        scheduler = False

    # need two image loss functions if bidirectional
    if bidir:
        losses = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses = [image_loss_func]
        weights = [1]

    losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
    weights += [args.weight]

    # Train

    # Set up names for checkpoints
    if args.image_loss == 'mse':
        ckpt = args.trained_model_fn +  '_mse_loss_vxm'
        checkpoint_path = os.path.join(args.trained_model_dir,
                                       args.trained_model_fn,
                                       ckpt + '.pth.tar')
    else:
        ckpt = args.trained_model_fn + '_ncc_loss'
        checkpoint_path = os.path.join(args.trained_model_dir,
                                       args.trained_model_fn,
                                       ckpt + '.pth.tar')
    if not os.path.exists(args.trained_model_dir):
        os.mkdir(args.trained_model_dir)

    # Set up TensorBoard writer
    if not args.log_dir:
        tb_dir = os.path.join(args.trained_model_dir, args.trained_model_fn + '_tb_logs')
    else:
        tb_dir = os.path.join(args.log_dir, args.trained_model_fn + '_tb_logs')

    logs_writer = SummaryWriter(tb_dir)

    # Start of training
    print('Starting training...')

    best_val_loss = float("inf")

    for epoch in range(1, args.num_epochs+1):

        # we don't need the average epoch loss so we assign it to _
        _ = train(epoch, model, losses, weights, optimizer,
                  dataloader, pair_generation_tnf,
                  log_interval=args.log_interval,
                  scheduler=scheduler,
                  tb_writer=logs_writer)

        val_loss = validate_model(model, losses, weights,
                                  dataloader_val, pair_generation_tnf,
                                  epoch, logs_writer)

        # remember best loss
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        save_checkpoint({
                         'epoch': epoch + 1,
                         'args': args,
                         'state_dict': model.state_dict(),
                         'best_val_loss': best_val_loss,
                         'optimizer': optimizer.state_dict(),
                         },
                        is_best, checkpoint_path)

    logs_writer.close()
    print('Done!')


if __name__ == '__main__':
    main()
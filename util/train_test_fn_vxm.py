from __future__ import print_function, division
import numpy as np
from tqdm import tqdm


def train(epoch, model, losses, weights, optimizer,
          dataloader, pair_generation_tnf,
          log_interval=50, tb_writer=None, scheduler=False):
    """
    Main function for training

    :param epoch: int, epoch index
    :param model: pytorch model object
    :param loss_fn: loss function of the model
    :param optimizer: optimizer of the model
    :param dataloader: DataLoader object
    :param pair_generation_tnf: Function to serve couples of samples
    :param log_interval: int, number of steps before logging scalars
    :param tb_writer: pytorch TensorBoard SummaryWriter
    :param scheduler: Eventual Learning rate scheduler

    :return: float, avg value of loss fn over epoch
    """

    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Epoch {}'.format(epoch))):
        optimizer.zero_grad()
        tnf_batch = pair_generation_tnf(batch)
        y_pred = model(tnf_batch)
        loss = 0
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(tnf_batch['target_image'], y_pred[n]) * weights[n]
            loss += curr_loss

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()
            if tb_writer:
                tb_writer.add_scalar('learning rate',
                                     scheduler.get_lr()[-1],
                                     (epoch - 1) * len(dataloader) + batch_idx)

        train_loss += loss.data.cpu().numpy().item()

        # log every log_interval
        if batch_idx % log_interval == 0:
            print('\tLoss: {:.6f}'.format(loss.data.item()))
            if tb_writer:
                tb_writer.add_scalar('training loss',
                                     loss.data.item(),
                                     (epoch - 1) * len(dataloader) + batch_idx)

    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def validate_model(model, losses, weights,
                   dataloader, pair_generation_tnf,
                   epoch, tb_writer=None):

    model.eval()
    val_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        tnf_batch = pair_generation_tnf(batch)
        y_pred = model(tnf_batch)

        loss = 0
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(tnf_batch['target_image'], y_pred[n]) * weights[n]
            loss += curr_loss

        val_loss += loss.data.cpu().numpy().item()

    val_loss /= len(dataloader)
    print('Validation set: Average loss: {:.4f}'.format(val_loss))
    if tb_writer:
        tb_writer.add_scalar('val loss',
                             val_loss,
                             epoch)

    return val_loss

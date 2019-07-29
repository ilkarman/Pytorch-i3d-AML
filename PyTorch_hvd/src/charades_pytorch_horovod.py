# See https://github.com/microsoft/DistributedDeepLearning/blob/pytorch_rev/%7B%7Bcookiecutter.project_name%7D%7D/PyTorch_imagenet/src/imagenet_pytorch_horovod.py

import logging
import logging.config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import fire
import shutil
from tqdm import tqdm


from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from azureml.core.run import Run


# Modules from folder
import videotransforms
from pytorch_i3d import InceptionI3d
from charades_dataset import Charades as Dataset


def _str_to_bool(in_str):
    if "t" in in_str.lower():
        return True
    else:
        return False


_MODE = "rgb"
_PRETRAINED_MODEL = "rgb_imagenet.pt"
_INIT_LR=0.01  # 0.1
_MAX_STEPS=64000
_BATCH_SIZE = 8  # per GPU
_NUM_CLASSES = 157
_SEED = 42
_DISTRIBUTED = _str_to_bool(os.getenv("DISTRIBUTED", "False"))

if _DISTRIBUTED:
    import horovod.torch as hvd

    hvd.init()


def _get_rank():
    if _DISTRIBUTED:
        try:
            return hvd.rank()
        except:
            return 0
    else:
        return 0


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        if _DISTRIBUTED:
            self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        else:
            self.sum += torch.sum(val.detach().cpu())
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


def _get_sampler(dataset, is_distributed=_DISTRIBUTED):
    if is_distributed:
        return torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank()
        )
    else:
        return torch.utils.data.sampler.RandomSampler(dataset)


def save_checkpoint(model, optimizer, filepath):
    if _get_rank() == 0:
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(state, filepath)


def train(gradient_update_steps, epoch, model, train_sampler, train_loader, optimizer, lr_sched, logger, 
    summary_writer=None, run=None):

    model.train()

    if _DISTRIBUTED:
        train_sampler.set_epoch(epoch)
    train_loc_loss = Metric('train_loc_loss')
    train_cls_loss = Metric('train_cls_loss')
    train_loss = Metric('train_loss')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1)) as t:

        for _, (data, target) in enumerate(train_loader):

            data, target = data.cuda(), target.cuda()
            n_frames = data.size(2)

            optimizer.zero_grad()
            # Returns: batch-size, num_classes, down-sampled-time steps
            per_frame_logits = model(data)           

            # Upsample output to input size 
            per_frame_logits = F.interpolate(
                per_frame_logits, n_frames, mode='linear', align_corners=True)

            # Charades dataset has classification & localisation tasks
            # Localisation loss 
            # find temporal locations of all activities in a video (per frame)
            loc_loss = F.binary_cross_entropy_with_logits(
                per_frame_logits, target)
            # compute classification loss (with max-pooling along time B x C x T)
            # recognise all activity categories for given videos (frames irrelevant)
            cls_loss = F.binary_cross_entropy_with_logits(
                torch.max(per_frame_logits, dim=2)[0], torch.max(target, dim=2)[0])

            # Total loss
            loss = 0.5*loc_loss + 0.5*cls_loss

            # Back-prop
            loss.backward() 
            optimizer.step()

            # Logging
            train_loc_loss.update(loc_loss)
            train_cls_loss.update(cls_loss)
            train_loss.update(loss)

            # Lr
            lr_sched.step()

            #t.set_postfix({
            #    'train_loc_loss': train_loc_loss.avg.item(),
            #    'train_cls_loss': train_cls_loss.avg.item(),
            #    'train_loss': train_loss.avg.item()
            #    })
            t.update(1)

            gradient_update_steps += 1

    # Write to tensorboard
    if summary_writer:
        summary_writer.add_scalar("train/loc_loss", train_loc_loss.avg, epoch)
        summary_writer.add_scalar("train/cls_loss", train_cls_loss.avg, epoch)
        summary_writer.add_scalar("train/loss", train_loss.avg, epoch)
    # Write to Azure
    if run:
        metrics = {'loc_loss': train_loc_loss.avg, 'cls_loss': train_cls_loss.avg, 'loss': train_loss.avg}
        run.log_row("Training metrics", epoch=epoch, **metrics)
    # Log to stdout
    #logger.info('Train: Avg Loc Loss: {:.4f} Avg Cls Loss: {:.4f} Avg Loss: {:.4f}'.format(
    #    train_loc_loss.avg,
    #    train_cls_loss.avg,
    #    train_loss.avg))

    return gradient_update_steps


def validate(epoch, model, val_loader, logger, 
    summary_writer=None, run=None):

    model.eval()

    val_loc_loss = Metric('val_loc_loss')
    val_cls_loss = Metric('val_cls_loss')
    val_loss = Metric('val_loss')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch     #{}'.format(epoch + 1)) as t:

        with torch.no_grad():
            for _, (data, target) in enumerate(val_loader):

                data, target = data.cuda(), target.cuda()
                n_frames = data.size(2)

                # Returns: batch-size, num_classes, down-sampled-time steps
                per_frame_logits = model(data)           

                # Upsample output to input size 
                per_frame_logits = F.interpolate(
                    per_frame_logits, n_frames, mode='linear', align_corners=True)

                # Charades dataset has classification & localisation tasks
                # Localisation loss 
                # find temporal locations of all activities in a video (per frame)
                loc_loss = F.binary_cross_entropy_with_logits(
                    per_frame_logits, target)
                # compute classification loss (with max-pooling along time B x C x T)
                # recognise all activity categories for given videos (frames irrelevant)
                cls_loss = F.binary_cross_entropy_with_logits(
                    torch.max(per_frame_logits, dim=2)[0], torch.max(target, dim=2)[0])

                # Total loss
                loss = 0.5*loc_loss + 0.5*cls_loss

                # Logging
                val_loc_loss.update(loc_loss)
                val_cls_loss.update(cls_loss)
                val_loss.update(loss)

                #t.set_postfix({
                #    'val_loc_loss': val_loc_loss.avg.item(),
                #    'val_cls_loss': val_cls_loss.avg.item(),
                #    'val_loss': val_loss.avg.item()
                #    })
                t.update(1)

    # Write to tensorboard
    if summary_writer:
        summary_writer.add_scalar("val/loc_loss", val_loc_loss.avg, epoch)
        summary_writer.add_scalar("val/cls_loss", val_cls_loss.avg, epoch)
        summary_writer.add_scalar("val/loss", val_loss.avg, epoch)
    # Write to azure
    if run:
        metrics = {'loc_loss': val_loc_loss.avg, 'cls_loss': val_cls_loss.avg, 'loss': val_loss.avg}
        run.log_row("Validation metrics", epoch=epoch, **metrics)
    # Log to stdout
    logger.info('Val: Avg Loc Loss: {:.4f} Avg Cls Loss: {:.4f} Avg Loss: {:.4f}'.format(
        val_loc_loss.avg,
        val_cls_loss.avg,
        val_loss.avg))


def main(
    root = "train/Charades_v1_rgb/",
    train_split="train/charades.json",
    use_gpu=False,
    save_folder='outputs'):

    # Create output/save_folder
    os.makedirs(save_folder, exist_ok=True)

    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if use_gpu else "cpu")
    logger.info(f"Running on {device}")
    if _DISTRIBUTED:
        # Horovod: initialize Horovod.

        logger.info("Running Distributed")
        torch.manual_seed(_SEED)
        if use_gpu:
            # Horovod: pin GPU to local rank.
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(_SEED)

    logger.info("PyTorch version {}".format(torch.__version__))

    # Horovod: write TensorBoard logs on first worker.
    if (_DISTRIBUTED and hvd.rank() == 0) or not _DISTRIBUTED:
        run = Run.get_context()
        run.tag("model", value="i3d")

        logs_dir = os.path.join(os.curdir, "logs")
        if os.path.exists(logs_dir):
            logger.debug(f"Log directory {logs_dir} found | Deleting")
            shutil.rmtree(logs_dir)
        
        summary_writer = SummaryWriter(logdir=logs_dir) 
    else:
        summary_writer = None
        run = None

    # CuDNN autotune
    torch.backends.cudnn.benchmark = True  # input size fixed

    # Define Dataset
    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip()])
    test_transforms = transforms.Compose([
        videotransforms.CenterCrop(224)])

    train_dataset = Dataset(train_split, 'training', root, _MODE, train_transforms)
    val_dataset = Dataset(train_split, 'testing', root, _MODE, test_transforms)

    # Partition dataset among workers using DistributedSampler
    train_sampler = _get_sampler(train_dataset)
    val_sampler = _get_sampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=_BATCH_SIZE,
        sampler=train_sampler, 
        num_workers=4,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=_BATCH_SIZE,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True)

    print("{} in training, {} in validation".format(len(train_dataset), len(val_dataset)))

    logger.info("Loading model")

    # Init model  
    if _PRETRAINED_MODEL == "rgb_imagenet.pt":
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load("{}/{}".format(root, _PRETRAINED_MODEL)))
        print("Loaded model weights for init")
        print(i3d)
        i3d.replace_logits(_NUM_CLASSES)
    else:
        i3d = InceptionI3d(_NUM_CLASSES, in_channels=3)

    if use_gpu:
        # Move model to GPU.
        i3d.cuda()

    # FLAG

    # Model not converging so try different LR methods from other repos
    lr = _INIT_LR
    # lr at the moment fixed (FLAG) depends on num of workers
    #optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    optimizer = optim.Adam(i3d.parameters(), lr=1e-4, weight_decay=1e-5)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    if _DISTRIBUTED:

        # Add Horovod Distributed Optimizer
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=i3d.named_parameters(),
            compression=hvd.Compression.none)  # alter later FLAG

        print('Broadcasting')
        # Broadcast parameters from rank 0 to all other processes.
        hvd.broadcast_parameters(i3d.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Accumulate gradient
    # https://github.com/piergiaj/pytorch-i3d/issues/20
    # "Our model was trained with 6 videos per GPU and 64 GPUs
    # (effective batch size of 384)."
    # Here we have 8 videos per GPU * 4 = 32 videos per VM
    # VM = 1, num_steps_per_update=12
    # VM = 4, num_steps_per_update=3
    # VM = 12, num_steps_per_update=1
    # Ideally scale to 12 VMs, with 4 GPUs and 8 videos per GPU

    epoch = 0
    gradient_update_steps = 0

    # Convergence was counted by gradient_update_steps where each step
    # Should contain around 384 samples
    while gradient_update_steps < _MAX_STEPS:
        logger.info('Step {}/{}, Epoch {}'.format(gradient_update_steps, _MAX_STEPS, epoch))
        # Train
        gradient_update_steps = train(
            gradient_update_steps, epoch, i3d, train_sampler, train_loader, optimizer, lr_sched, logger, run)
        # Validate
        validate(
            epoch, i3d, val_loader, logger, run)
        epoch += 1
    
    # Save
    if save_folder is not None:
        sname = os.path.join(save_folder,'charades'+str(gradient_update_steps).zfill(6)+'.pt')
        logger.info('Saving {}'.format(sname))
        save_checkpoint(
            i3d,
            optimizer, 
            sname)

    summary_writer.close()

if __name__ == "__main__":
    logging.config.fileConfig(os.getenv("LOG_CONFIG", "logging.conf"))
    fire.Fire(main)
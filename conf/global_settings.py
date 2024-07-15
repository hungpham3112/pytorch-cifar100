""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cangjie dataset
CIFAR100_TRAIN_MEAN = (0.5)
CIFAR100_TRAIN_STD = (0.5)

CIFAR100_TEST_MEAN = (0.5)
CIFAR100_TEST_STD = (0.5)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 6
MILESTONES = [2, 4, 6]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10









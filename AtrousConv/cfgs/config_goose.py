import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.name = 'goose'

cfg.hflip = False

# class_num should include background
cfg.class_num = 4

cfg.dilations = [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32]
cfg.channels = [32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128]
cfg.kernel_size = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
cfg.with_bn = True

cfg.weight_decay = 5e-4


cfg.gamma = 2

cfg.train_list = [cfg.name + "_train.txt"]
cfg.test_list = cfg.name + "_test.txt"
cfg.num_epochs = 150
cfg.batch_size = 1

cfg.cuda = True
cfg.seed = 22222
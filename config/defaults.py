from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '1'
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.DATA = ''

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 0

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 1e-4
# Batch size
_C.SOLVER.SEQUENCE_PER_BATCH = 32
_C.SOLVER.ITERS_TO_ACCUMULATE = 1
_C.SOLVER.M0MENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.NORM_WEIGHT_DECAY = 0.0
_C.SOLVER.LR_WARMUP_EPOCHS = 5
_C.SOLVER.LR_WARMUP_DECAY = 0.01

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.SEQUENCE_PER_BATCH = 32
# Path to trained model
_C.TEST.WEIGHT = ""
_C.TEST.EVALUATE_ONLY = 'off'
_C.TEST.DATA = ''

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
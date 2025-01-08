import random
import os
import numpy as np
import paddle
import logging
import argparse

from cd_models.fccdn import FCCDN
from cd_models.stanet import STANet
from cd_models.p2v import P2V
from cd_models.msfgnet import MSFGNet
from cd_models.fc_siam_conc import FCSiamConc
from cd_models.dsamnet import DSAMNet
from cd_models.snunet import SNUNet
from cd_models.f3net import F3Net
from paddleseg.models import UNet
from cd_models.replkcd import CD_RLKNet

from models.model import SCDSam, SCDSamV1, SCDSam_Mamba, SCDSam_Mambav1

from core.datasets.scdloader import MusReader
from core.scdmisc.predict import predict


dataset_name = "MusSCD"


dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

    
if __name__ == "__main__":
    print("main")
    
    dataset = MusReader(dataset_path, mode='val')
    model = SCDSam(256, 5)
    weight_path = r"/home/jq/Code/paddle/output/musscd/SCDSam_2024_12_05_12/last_model.pdparams"
    predict(model, dataset, weight_path, dataset_name, 5)
    



""" ASVspoof5 + ASVspoof2019 모댈로 DFADD, ASVSpoof2019, ASVSpoof2021, ASVSpoof5 평가"""

import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (genSpoof_list, genSpoof_list_DFADD, genSpoof_list_asv5)
from eval.calculate_metrics import calculate_minDCF_EER_CLLR, calculate_aDCF_tdcf_tEER
from utils import create_optimizer, seed_worker, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm
import soundfile as sf

#Seed 설정
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

def main():
    '''    -> 기본 모델
    MODEL_CONFIG = {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }
    '''
    #AASIST_L
    MODEL_CONFIG = {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
        "gat_dims": [24, 32],
        "pool_ratios": [0.4, 0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }

    """    MODEL LOAD    """ 
    #/data/kdc/aasist/models/weights/AASIST.pth -> 2019 model
    model_path = "../../../data/kdc/aasist/models/weights/AASIST-L.pth" #"./exp_result/AASIST_ASVspoof5_ep100_bs32/weights/epoch_55_0.250.pth" #./exp_result/AASIST_ASVspoof5_ep100_bs32/weights/epoch_55_0.250.pth  ./exp_result/Finetune/weights/epoch_99_0.006.pth /app/asvspoof5/Baseline-AASIST/exp_result/AASIST_ASVspoof5_ep100_bs32/weights/epoch_55_0.250.pth
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(MODEL_CONFIG, device, model_path)

    print("Model loaded from {}".format(model_path))
    print("Device:",device)  

    output_dir = "./exp_result/2019_5/" #Evaluation_all
    output_dir = Path(output_dir)
    print("Results will be saved to {}".format(output_dir))

    #Evaluation for DFADD
    print("DFADD 평가 시작")
    eval_DFADD(model, device, output_dir)
    print("DFADD 평가 완료")
    print("ASVSpoof5 평가 시작")
    eval_asv5(model, device, output_dir)
    print("ASVSpoof5 평가 완료")
    print("ASVSpoof2019 평가 시작")
    eval_asv2019(model, device, output_dir)
    print("ASVSpoof2019 평가 완료")



def get_model(model_config: Dict, device: torch.device, model_path):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    model.load_state_dict(
        torch.load(model_path, map_location=device))


    return model

def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str,
    asv5 = False) -> None:

    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    #assert len(trial_lines) == len(fname_list) == len(score_list)
    if asv5:
        with open(save_path, "w") as fh:
            for fn, sco, trl in zip(fname_list, score_list, trial_lines):
                spk_id, utt_id, _, _, _, _, _, src, key, _ = trl.strip().split(' ')
                assert fn == utt_id
                fh.write("{} {} {} {}\n".format(spk_id, utt_id, sco, key))
        print("Scores saved to {}".format(save_path))
    else:
        with open(save_path, "w") as fh:
            for fn, sco, trl in zip(fname_list, score_list, trial_lines):
                spk_id, utt_id, _, _, key= trl.strip().split(' ')
                assert fn == utt_id
                fh.write("{} {} {} {}\n".format(spk_id, utt_id, sco, key))
        print("Scores saved to {}".format(save_path))

    
def eval_DFADD(model, device, output_dir):

    """TEST DATASET CLASS for DFADD"""
    class TestDataset_DFADD(Dataset):
        def __init__(self, list_IDs, base_dir, dataset_name):
            """self.list_IDs	: list of strings (each string: utt key),
            """
            self.list_IDs = list_IDs
            self.base_dir = Path(base_dir) #../../../data/DFADD/DFADD_ZIP/DATASET_GradTTS/test
            self.cut = 64600  # take ~4 sec audio (64600 samples)
            self.dataset_name = dataset_name #'GradTTS', 'matcha', 'NaturalSpeech2', 'pflow', 'StyleTTS2'

        def __len__(self):
            return len(self.list_IDs)

        def __getitem__(self, index):
            key = self.list_IDs[index]
            if self.dataset_name in key: 
                X, _ = sf.read(str(Path(self.base_dir) / f"{key}.flac"))
            else: #bonafide의 경우
                bonafide_dir = self.base_dir.parent.parent / "DATASET_VCTK_BONAFIDE/test"
                X, _ = sf.read(str(Path(bonafide_dir) / f"{key}.wav"))

            X_pad = pad(X, self.cut)
            x_inp = Tensor(X_pad)
            return x_inp, key


    DATA_PATH = Path("../../../data/DFADD/DFADD_ZIP")
    SUBSET_MAPPING = { #key: [dir_name, dataset_short_name]
    'D1': ['DATASET_GradTTS','GradTTS'],
    'D2': ['DATASET_MatchaTTS', 'matcha'],
    'D3': ['DATASET_NaturalSpeech2', 'NaturalSpeech2'],
    'F1': ['DATASET_PflowTTS', 'pflow'],
    'F2': ['DATASET_StyleTTS2', 'StyleTTS2'],
    }
    for key, value in SUBSET_MAPPING.items(): #key : D1 / value : [DATASET_GradTTS, 'GradTTS']
        print(f"Processing subset {key} - {value}") 
        subset_output_dir = output_dir / key
        subset_output_dir.mkdir(parents=True, exist_ok=True)

        data_path = DATA_PATH / value[0] / "test" #flac file dir
        txt_path = DATA_PATH / value[0] / "test.txt" #meta file dir

        d_meta, file_list = genSpoof_list_DFADD(str(txt_path), is_train = False, is_eval = False)

        test_dataset = TestDataset_DFADD(list_IDs = file_list, base_dir = str(data_path), dataset_name = value[1] ) #TESTDATASET 변경 필요.
        test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False,\
                            drop_last = False, num_workers = 4, pin_memory = True)

        produce_evaluation_file(test_loader, model, device, \
                                save_path = Path(subset_output_dir / "scores.txt"), \
                                trial_path = Path(txt_path) )
        
        # Calculate metrics
        eval_dcf, eval_eer, eval_cllr = calculate_minDCF_EER_CLLR(
            cm_scores_file = subset_output_dir/"scores.txt",
            output_file = subset_output_dir/"metrics.txt")
        print(f"{key} - {value} : eval_eer: {eval_eer:.3f}, eval_dcf:{eval_dcf:.5f} , eval_cllr:{eval_cllr:.5f}")


def eval_asv5(model, device, output_dir):

    DATA_PATH = Path("../../../data/ASVspoof/ASVspoof5")
    dev_protocol = DATA_PATH / "ASVspoof5.eval.track_1.tsv"
    dev_path = DATA_PATH / "flac_E_eval"             # eval / dev/

    output_dir = output_dir / "ASV5"
    output_dir.mkdir(parents = True, exist_ok = True)


    d_dev, file_list_dev = genSpoof_list_asv5(dir_meta = str(dev_protocol), is_train = False, is_eval = False)
    dev_dataset = TestDataset(list_IDs = file_list_dev[:2000], base_dir = str(dev_path))
    dev_loader = DataLoader(dev_dataset, batch_size = 32, shuffle = False,\
                            drop_last = False, num_workers = 4, pin_memory = True)

    produce_evaluation_file(dev_loader, model, device, \
                            save_path = Path(output_dir / "scores_dev.txt"), \
                            trial_path = Path(dev_protocol), asv5 = True )
    eval_dcf, eval_eer, eval_cllr = calculate_minDCF_EER_CLLR(
        cm_scores_file = output_dir/"scores_dev.txt",
        output_file = output_dir/"metrics_dev.txt")
    print(f"ASVspoof5_dev : eval_eer: {eval_eer:.3f}, eval_dcf:{eval_dcf:.5f} , eval_cllr:{eval_cllr:.5f}")



def eval_asv2019(model, device, output_dir):
    
    DATA_PATH = Path("../../../data/ASVspoof/ASVspoof2019/LA") #ASVspoof2019 LA 
    cm_protocol = DATA_PATH / "ASVspoof2019_LA_cm_protocols"
    dev_path = DATA_PATH / "ASVspoof2019_LA_eval/flac"             # eval / dev/
    dev_protocol = cm_protocol / "ASVspoof2019.LA.cm.eval.trl.txt" # eval / dev/

    d_dev, file_list_dev = genSpoof_list_DFADD(str(dev_protocol), is_train = False, is_eval = False)
    dev_dataset = TestDataset(list_IDs = file_list_dev, base_dir = str(dev_path))
    dev_loader = DataLoader(dev_dataset, batch_size = 32, shuffle = False,\
                            drop_last = False, num_workers = 4, pin_memory = True)

    output_dir = output_dir / "ASV2019"
    output_dir.mkdir(parents=True, exist_ok=True)

    produce_evaluation_file(dev_loader, model, device, \
                            save_path = Path(output_dir / "scores_dev.txt"), \
                            trial_path = Path(dev_protocol) )
    eval_dcf, eval_eer, eval_cllr = calculate_minDCF_EER_CLLR(
        cm_scores_file = output_dir/"scores_dev.txt",
        output_file = output_dir/"metrics_dev.txt")
    print(f"ASVspoof2019_dev : eval_eer: {eval_eer:.3f}, eval_dcf:{eval_dcf:.5f} , eval_cllr:{eval_cllr:.5f}")


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x



class TestDataset(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = Path(base_dir)
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(Path(self.base_dir) / f"{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key

if __name__ == "__main__":
    main()

"""
AASIST에 ASVspoof5 + ASVspoof2019 종합 데이터셋 훈련을 위한 코드입니다.
공식 github의 ASVspoof5 pretrained weight에 2019 데이터셋을 파인튜닝 하는 방식입니다.
main 함수의 DATA_PATH를 2019 LA 데이터셋 경로로 설정해주시면 됩니다.
"""
import argparse
import json
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

from data_utils import (genSpoof_list_asv5, genSpoof_list_DFADD) #DFADD랑 2019랑 형식이 같아서 DFADD 사용.
from eval.calculate_metrics import calculate_minDCF_EER_CLLR, calculate_aDCF_tdcf_tEER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm
import soundfile as sf

#Seed 설정

def main(args: argparse.Namespace) -> None:

    DATA_PATH = Path("../../../data/ASVspoof") #ASVspoof2019 LA 
    DATA_PATH_2019 = DATA_PATH / "ASVspoof2019/LA"
    DATA_PATH_5 = DATA_PATH / "ASVspoof5"
    
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())

    set_seed(args.seed, config)
    
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]

    """    MODEL LOAD    """
    model_path = "./models/weights/AASIST/best.pth" #--> spoof5 데이터셋에 학습된 모델 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_config, device)
    model.load_state_dict(
        torch.load(model_path, map_location=device))

    print("Model loaded from {}".format(model_path))
    print("Device:",device)  

    """Fine-tuning을 위한 데이터 경로 설정"""

    output_dir = "./exp_result/Finetune"
    output_dir = Path(output_dir) #./exp_result/finetune 로 설정.
    print("Results will be saved to {}".format(output_dir))

    cm_protocol_2019 = DATA_PATH_2019 / "ASVspoof2019_LA_cm_protocols"
    train_path = DATA_PATH_2019 / "ASVspoof2019_LA_train/flac"
    train_protocol = cm_protocol_2019 / "ASVspoof2019.LA.cm.train.trn.txt"

    dev_path_2019 = DATA_PATH_2019 / "ASVspoof2019_LA_dev/flac"
    dev_protocol_2019 = cm_protocol_2019 / "ASVspoof2019.LA.cm.dev.trl.txt"

    dev_path_5 = DATA_PATH_5 / "flac_D/"
    dev_protocol_5 = DATA_PATH_5 / "ASVspoof5.dev.track_1.tsv"

    model_dir = output_dir / "weights"
    os.makedirs(model_dir, exist_ok = True)


    """    DATA LOADERS   """

    train_loader, dev_loader_2019, dev_loader_5 = get_loader(train_path, train_protocol, dev_path_2019, dev_protocol_2019, dev_path_5, dev_protocol_5)
    print("Data loaders ready.")

    """    OPTIMIZER    """

    optim_config["steps_per_epoch"] = len(train_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 100.
    best_dev_dcf = 1.
    best_dev_cllr = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    metric_path = output_dir/"metrics"
    os.makedirs(metric_path, exist_ok = True)
    copy(args.config, output_dir / "config.json")

    """   TRAINING   """
    print("Start training...")
    n2019 = len(dev_loader_2019.dataset)
    n5    = len(dev_loader_5.dataset)

    w2019 = n2019 / (n2019 + n5)
    w5    = n5    / (n2019 + n5)

    for epoch in range(config["num_epochs"]):
        print("training epoch{:03d}".format(epoch))
        
        running_loss = train_epoch(train_loader, model, optimizer, device,
                                   scheduler, config)
        
        produce_evaluation_file(dev_loader_2019, model, device,
                                metric_path/"dev_score.txt", dev_protocol_2019)
        produce_evaluation_file(dev_loader_5, model, device,
                                metric_path/"dev_score_asv5.txt", dev_protocol_5, asv5 = True)


        dev_dcf_2019, dev_eer_2019, dev_cllr_2019 = calculate_minDCF_EER_CLLR(
            cm_scores_file=metric_path/"dev_score.txt",
            output_file=metric_path/"dev_DCF_EER_{}epo.txt".format(epoch),
            printout=False)


        dev_dcf_asv5, dev_eer_asv5, dev_cllr_asv5 = calculate_minDCF_EER_CLLR(
            cm_scores_file=metric_path/"dev_score_asv5.txt",
            output_file = metric_path/"dev_DCF_EER_asv5_{}epo.txt".format(epoch),
            printout=False
        )

        dev_eer = w2019 * dev_eer_2019 + w5 * dev_eer_asv5
        dev_dcf = w2019 * dev_dcf_2019 + w5 * dev_dcf_asv5
        dev_cllr = w2019 * dev_cllr_2019 + w5 * dev_cllr_asv5


        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_dcf:{:.5f} , dev_cllr:{:.5f}".format(
            running_loss, dev_eer, dev_dcf, dev_cllr))
        print("  --> ASVspoof2019 LA dev eer: {:.3f}, dcf:{:.5f}, cllr:{:.5f}".format(
            dev_eer_2019, dev_dcf_2019, dev_cllr_2019)
        )
        print("  --> ASVspoof5 dev eer: {:.3f}, dcf:{:.5f}, cllr:{:.5f}".format(
            dev_eer_asv5, dev_dcf_asv5, dev_cllr_asv5)
        )

        torch.save(model.state_dict(),
                    model_dir / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))


        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            best_dev_dcf = dev_dcf
            best_dev_cllr = dev_cllr
            torch.save(
            model.state_dict(),
            model_dir / "best_finetuned.pth"
            )  

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1



def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(train_path, train_protocol, dev_path_2019, dev_protocol_2019, dev_path_5, dev_protocol_5):
    d_trn, file_list_trn = genSpoof_list_DFADD(str(train_protocol), is_train = True, is_eval = False)
    d_dev, file_list_dev = genSpoof_list_DFADD(str(dev_protocol_2019), is_train = False, is_eval = False)

    d_dev_5, file_list_dev_5 = genSpoof_list_asv5(str(dev_protocol_5), is_train = False, is_eval = False)
    dev_dataset_asv5 = TestDataset(list_IDs = file_list_dev_5[:2000], base_dir = str(dev_path_5))
    dev_loader_asv5 = DataLoader(dev_dataset_asv5, batch_size = 32, shuffle = False,\
                            drop_last = False, num_workers = 4, pin_memory = True)


    trn_dataset = TrainDataset(list_IDs = file_list_trn, labels = d_trn, base_dir = str(train_path))
    dev_dataset = TestDataset(list_IDs = file_list_dev, base_dir = str(dev_path_2019))
    gen = torch.Generator()
    gen.manual_seed(0)

    train_loader = DataLoader(trn_dataset, batch_size = 32, shuffle = True,\
                            drop_last = True, num_workers = 4, pin_memory = True, worker_init_fn=seed_worker, generator=gen)
    dev_loader = DataLoader(dev_dataset, batch_size = 32, shuffle = False,\
                            drop_last = False, num_workers = 4, pin_memory = True)

    return train_loader, dev_loader, dev_loader_asv5


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


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in tqdm(trn_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss



def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class TrainDataset(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = Path(base_dir)
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


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
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())

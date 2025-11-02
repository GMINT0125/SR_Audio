import numpy as np
import soundfile as sf
import torch
from torch import Tensor
import torchaudio
from torchaudio.functional import apply_codec
from torch.utils.data import Dataset
from pathlib import Path


___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _, _ = line.strip().split(" ")
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

def genSpoof_list_DFADD(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def genSpoof_list_asv5(dir_meta, is_train = False, is_eval = False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, _, _, attach_cat, attack_type, label, _ = line.strip().split(" ") #cat : AC1, AC2, AC3
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _, _, attach_cat, attack_type, label, _ = line.strip().split(" ")
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, _, _, attach_cat, attack_type, label, _ = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

def get_attack_type(dir_meta, is_train = False, is_eval = False):

    d_meta = {}
    file_list = []
    atk_cat = []
    atk_type = []

    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, _, _, attack_cat, attack_type, label, _ = line.strip().split(" ") #cat : AC1, AC2, AC3
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
            atk_cat.append(attack_cat)
            atk_type.append(attack_type)

        return atk_cat, atk_type

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _, _, attack_cat, attack_type, label, _ = line.strip().split(" ")
            file_list.append(key)
            atk_cat.append(attack_cat)
            atk_type.append(attack_type)
        return atk_cat, atk_type
    else:
        for line in l_meta:
            _, key, _, _, _, _, attack_cat, attack_type, label, _ = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
            atk_cat.append(attack_cat)
            atk_type.append(attack_type)
        return atk_cat, atk_type


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
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"{key}.flac"))  #tensor로 바꿔서 전처리 적용해야 한다.
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

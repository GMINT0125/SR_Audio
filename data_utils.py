"""
여러 데이터셋에 맞는 데이터 로딩 유틸리티 함수 및 클래스입니다.
"""
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
import torchaudio
from torchaudio.functional import apply_codec
from torch.utils.data import Dataset
from pathlib import Path
from util.RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav


___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    """
    기본 메타데이터 파싱 함수
    DFADD / ASVspoof5, 2019 데이터셋은 별도 메타데이터 파싱 함수 구현
    """
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
    """
    DFADD 메타데이터 파싱 
    : 2019의 경우 DFADD와 메타데이터 형식이 동일하여 동일한 함수를 사용합니다.

    d_meta : 딕셔너리 (key: utt key, value: label integer)
    file_list : 리스트 (utt key 문자열 리스트)
    """
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
    """
    ASVspoof5 메타데이터 파싱 함수 
    """
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, _, _, attach_cat, attack_type, label, _ = line.strip().split(" ")
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
    """
    util/visualize_embed.py 함수의 ASVspoof5 공격 유형 시각화를 위한 함수
    메타데이터를 파싱해서 attack category 및 attack type 리스트를 반환합니다.
    """

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
    def __init__(self, list_IDs, labels, base_dir, args):
        """
        self.list_IDs	: list of strings (each string: utt key),
        self.labels      : dictionary (key: utt key, value: label integer)
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.codec = args.codec

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        base_dir = self.base_dir
        if self.codec:
            if np.random.rand() < 0.2:
                base_dir = Path(str(base_dir) + "_codec")
            X, _ = sf.read(str(base_dir / f"{key}.flac")) 
            X_pad = pad_random(X, self.cut)
            x_inp = Tensor(X_pad)
            y = self.labels[key]
            return x_inp, y

        else:
            X, _ = sf.read(str(base_dir / f"{key}.flac"))
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

class  TrainDatasetWithRawBoost(Dataset):
    def __init__(self, list_IDs, labels, base_dir, algo, args):
        """
        RawBoost 증강 적용한 TrainDataset 클래스 
        self.list_IDs	: list of strings (each string: utt key),
        self.labels      : dictionary (key: utt key, value: label integer)
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        # rawboost
        self.algo=algo
        self.args=args
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.codec = args.codec

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        base_dir = self.base_dir
        if self.codec:
            if np.random.rand() < 0.2:
                base_dir = Path(str(base_dir) + "_codec")
            x, sr = sf.read(str(base_dir / f"{key}.flac"))
            x = x.astype(np.float32, copy=False)
            x = pad_random(x, self.cut)                 # 4초 먼저
            Y = process_Rawboost_feature(x, sr, self.args, self.algo)  # 여기에만 RawBoost
            x_inp = Tensor(Y)
            y = self.labels[key]
            return x_inp, y
        else:
            x, sr = sf.read(str(base_dir / f"{key}.flac"))
            x = x.astype(np.float32, copy=False)
            x = pad_random(x, self.cut)                 # 4초 먼저
            Y = process_Rawboost_feature(x, sr, self.args, self.algo)  # 여기에만 RawBoost
            x_inp = Tensor(Y)
            y = self.labels[key]
            return x_inp, y

def process_Rawboost_feature(feature, sr, args, algo):
    """
    RawBoost 증강 알고리즘 
    1 : LnL convolutive noise
    2 : ISD impulsive noise
    3 : SSI coloured additive noise
    """
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        feature=feature
    
    return feature

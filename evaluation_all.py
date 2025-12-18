"""
데이터셋들에 대하여 학습된 모델을 평가하는 코드입니다.
평가 가능 Dataset =
[   
    1. ASVspoof 2019
    2. ASVspoof 5 (2024)
    3. DFADD 
    4. Unseen (elevenlabs, gemini, openai)
]   
1. 평가가 필요한 가중치 경로를 main 함수의 model_path를 변경해주세요.
2. eval_{dataset} 함수 내부 DATA_PATH를 본인의 데이터셋 경로로 변경 후 실행하면 됩니다. 
"""

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
import librosa

#Seed 설정
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

def main():
    MODEL_CONFIG = {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }
    """    MODEL LOAD    """ 
    model_path = "./models/weights/AASIST/best.pth" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(MODEL_CONFIG, device, model_path)

    print("Model loaded from {}".format(model_path))
    print("Device:",device)  

    output_dir = "./exp_result/"
    output_dir = Path(output_dir)
    print("Results will be saved to {}".format(output_dir))

    print("DFADD 평가 시작")
    eval_DFADD(model, device, output_dir)
    print("DFADD 평가 완료")

    print("ASVspoof2019 평가 시작")
    eval_asv2019(model, device, output_dir)
    print("ASVspoof2019 평가 완료")

    print("ASVspoof5 평가 시작")
    eval_asv5(model, device, output_dir)
    print("ASVspoof5 평가 완료")
    
    print("Unseen Dataset 평가 시작")
    eval_unseen(model, device, output_dir)
    print("Unseen 평가 완료")

def get_model(model_config: Dict, device: torch.device, model_path):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))


    """가중치 로드"""
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

def produce_evaluation_unseen(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: Union[str, Path]
) -> None:
    """
    Save format:
      spk_id(Unseen)  utt_id  score  key
    """
    model.eval()
    save_path = Path(save_path)

    with open(save_path, "w") as fh:
        for pack in tqdm(data_loader):
            if pack is None:
                continue
            batch_x, utt_ids, keys = pack

            batch_x = batch_x.to(device)

            with torch.no_grad():
                _, batch_out = model(batch_x)
                batch_score = batch_out[:, 1].data.cpu().numpy().ravel()

            for uid, sco, key in zip(utt_ids, batch_score, keys):
                fh.write(f"Unseen {uid} {sco:.6f} {key}\n")

    print(f"[Unseen] Scores saved to {save_path}")

def eval_DFADD(model, device, output_dir):

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
    dev_path = DATA_PATH / "flac_E_eval"             

    output_dir = output_dir / "ASV5"
    output_dir.mkdir(parents = True, exist_ok = True)


    d_dev, file_list_dev = genSpoof_list_asv5(dir_meta = str(dev_protocol), is_train = False, is_eval = False)
    dev_dataset = TestDataset(list_IDs = file_list_dev[:2000], base_dir = str(dev_path))
    dev_loader = DataLoader(dev_dataset, batch_size = 32, shuffle = False,\
                            drop_last = False, num_workers = 4, pin_memory = True)

    produce_evaluation_file(dev_loader, model, device, \
                            save_path = Path(output_dir / "scores_dev.txt"), \
                            trial_path = Path(dev_protocol), asv5 = True)
    
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

def eval_unseen(model, device, output_dir):
    """
    Unseen Dataset 평가 코드
    bonafide는 DFADD의 VCTK 데이터를 사용하여 정량 평가
    3가지 경우에 대해서 평가를 진행합니다.

        1. Bonafide + Cloning + TTS
        2. Bonafide + Cloning
        3. Bonafide + 개별 TTS 

    output은 exp_result의 Unseen 폴더에 저장됩니다.

    """
    DATA_PATH = Path("../../../data/UnseenTTS")
    spoof_clone = DATA_PATH / "cloning"
    spoof_tts   = DATA_PATH / "tts"
    bonafide    = DATA_PATH.parent / "DFADD/DFADD_ZIP/DATASET_VCTK_BONAFIDE/test"

    valid_ext = {"wav", "flac", "mp3"}

    # ----------------------------
    # 1) bonafide 리스트
    # ----------------------------
    file_list_bonafide = []
    for fn in os.listdir(bonafide):
        if fn.lower().endswith(".wav"):
            utt_id = Path(fn).stem
            file_list_bonafide.append((utt_id, "bonafide"))

    # ----------------------------
    # 2) cloning 리스트 (전체만 필요)
    # ----------------------------
    file_list_cloning = []
    for folder in os.listdir(spoof_clone):
        folder_path = spoof_clone / folder
        if not folder_path.is_dir():
            continue
        for fn in os.listdir(folder_path):
            ext = Path(fn).suffix.lower().lstrip(".")
            if ext in valid_ext:
                utt_id = Path(fn).stem
                file_list_cloning.append((folder, utt_id, "clone", "spoof"))

    # ----------------------------
    # 3) tts 리스트 (폴더별로 쪼개서 필요)
    #    tts_by_folder[folder] = [(folder, utt_id, "tts","spoof"), ...]
    # ----------------------------
    tts_by_folder = {}
    for folder in os.listdir(spoof_tts):
        folder_path = spoof_tts / folder
        if not folder_path.is_dir():
            continue

        cur_list = []
        for fn in os.listdir(folder_path):
            ext = Path(fn).suffix.lower().lstrip(".")
            if ext in valid_ext:
                utt_id = Path(fn).stem
                cur_list.append((folder, utt_id, "tts", "spoof"))

        if len(cur_list) > 0:
            tts_by_folder[folder] = cur_list

    # ----------------------------
    # 4) output root
    # ----------------------------
    out_root = Path(output_dir) / "Unseen"
    out_root.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 5) 케이스별 실행 함수
    # ----------------------------
    def run_case(case_name: str, file_list_case):
        case_dir = out_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        test_dataset = TestDataset_Unseen(
            list_IDs=file_list_case,
            audio_dirs=[spoof_clone, spoof_tts, bonafide],
            cut=64600
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_drop_none
        )

        score_path = case_dir / "scores.txt"
        produce_evaluation_unseen(test_loader, model, device, score_path)

        eval_dcf, eval_eer, eval_cllr = calculate_minDCF_EER_CLLR(
            cm_scores_file=score_path,
            output_file=case_dir / "metrics.txt"
        )
        print(f"[{case_name}] eval_eer: {eval_eer:.3f}, eval_dcf:{eval_dcf:.5f}, eval_cllr:{eval_cllr:.5f}")

    # ----------------------------
    # 6) (1) ALL
    # ----------------------------
    all_tts = []
    for lst in tts_by_folder.values():
        all_tts.extend(lst)
    run_case("ALL", file_list_bonafide + file_list_cloning + all_tts)

    # ----------------------------
    # 7) (2) CLONING only + bonafide
    # ----------------------------
    run_case("CLONING", file_list_bonafide + file_list_cloning)

    # ----------------------------
    # 8) (3) TTS 폴더별 + bonafide
    # ----------------------------
    for folder_name, tts_list in sorted(tts_by_folder.items()):
        # 폴더명이 너무 길거나 슬래시 있으면 안전하게 바꿔도 됨(필요시)
        run_case(folder_name, file_list_bonafide + tts_list)

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

class TestDataset_DFADD(Dataset):
    def __init__(self, list_IDs, base_dir, dataset_name):
        """
        DFADD test의 경우, ASVSpoof와 확장자가 일치하지 않기 떄문에 별도의 TestDatset 클래스 구현
        self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = Path(base_dir) # ex) ../../../data/DFADD/DFADD_ZIP/DATASET_GradTTS/test
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

class TestDataset_Unseen(Dataset):
    def __init__(self, list_IDs, audio_dirs, cut=64600):
        """
        Unseen Dataset 전용 TestDataset 클래스
        list_IDs:
          bonafide: (utt_id, "bonafide")
          spoof:    (folder_name, "utt_id", "clone/tts", "spoof")
        audio_dirs: [spoof_clone_dir, spoof_tts_dir, bonafide_dir] (Path or str)
        """
        self.list_IDs = list_IDs
        self.spoof_clone_dir = Path(audio_dirs[0])
        self.spoof_tts_dir   = Path(audio_dirs[1])
        self.bonafide_dir    = Path(audio_dirs[2])
        self.cut = cut

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        meta = self.list_IDs[index]
        tts_set = {"elevenlabs" : "mp3",
                "gemini": "wav",
                "openai": "flac"}
        try:
            if meta[-1] == "bonafide":
                utt_id, key = meta
                audio_path = self.bonafide_dir / f"{utt_id}.wav"
                unique_id = f"bonafide__{utt_id}"

            else:  # spoof
                folder_name, utt_id, atktype, key = meta
                if atktype == "clone":
                    audio_path = self.spoof_clone_dir / folder_name / f"{utt_id}.wav"
                else:
                    ext = None
                    for tts, file_ext in tts_set.items():
                        if tts in folder_name.lower():
                            ext = file_ext
                            break
                    if ext is None:
                        raise RuntimeError(f"Unknown TTS folder: {folder_name}")
                    audio_path = self.spoof_tts_dir / folder_name / f"{utt_id}.{ext}"
                unique_id = f"{folder_name}__{utt_id}"
            
            if audio_path.suffix.lower() == ".mp3":
                # mp3는 librosa )
                X, _ = librosa.load(
                    str(audio_path),
                    sr=16000,
                    mono=True,
                    duration=self.cut / 16000
                )
            else:
                # wav / flac는 SoundFile
                with sf.SoundFile(str(audio_path)) as f:
                    max_frames = int((self.cut / 16000) * f.samplerate)
                    max_frames = max(1, max_frames)
                    X = f.read(
                        frames=max_frames,
                        dtype="float32",
                        always_2d=False
                    )
                    if X.ndim == 2:
                        X = X.mean(axis=1)
                    if f.samplerate != 16000:
                        X = librosa.resample(
                            y=X,
                            orig_sr=f.samplerate,
                            target_sr=16000
                        )

            # ---------------------------
            # 3. pad & return
            # ---------------------------
            X_pad = pad(X, self.cut)
            return Tensor(X_pad), unique_id, key

        except Exception as e:
            print(f"[DROP] {meta} | {e}")
            return None
def collate_drop_none(batch):
    """파일이 깨졌을 경우 None이 리턴될 수 있으므로 이를 제거하는 collate 함수"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    xs, uids, keys = zip(*batch)
    return torch.stack(xs), list(uids), list(keys)

if __name__ == "__main__":
    main()

"""Visualize aasist-embed using t-SNE or some other method (dataset: ASVspoof5)"""

"""step 1: extract embeddings from aasist-model (Train data?)"""
"""step 2: visualize embeddings using t-SNE"""


import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (genSpoof_list, genSpoof_list_DFADD, genSpoof_list_asv5, get_attack_type)
from eval.calculate_metrics import calculate_minDCF_EER_CLLR, calculate_aDCF_tdcf_tEER
from evaluation_all import get_model, TestDataset
from utils import create_optimizer, seed_worker, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm
import soundfile as sf

def main(data_type):

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

    model_path = "./exp_result/AASIST_ASVspoof5_ep100_bs32/weights/epoch_55_0.250.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(MODEL_CONFIG, device, model_path)

    print("Model loaded from {}".format(model_path))
    print("Device:",device)  


    all_embeddings, all_labels, attack_category, attack_type = extract_embedding(model ,device, mode = 'asv5', data_type = data_type)
    visualize_tsne(all_embeddings, all_labels, attack_category, attack_type, data_type = data_type)

def extract_embedding(model, device, mode = 'asv5', data_type = 'dev'):

    if mode == 'asv5':
        DATA_PATH = Path("../../../data/ASVspoof/ASVspoof5")
        dev_protocol = DATA_PATH / "ASVspoof5.dev.track_1.tsv"
        dev_path = DATA_PATH / "flac_D"

        d_dev, file_list_dev = genSpoof_list_asv5(dir_meta = str(dev_protocol), is_train = False, is_eval = False)
        

        #d_dev == label // 1 : bonafide, 0 : spoof
        dev_dataset = TestDataset(list_IDs = file_list_dev[:2000], base_dir = str(dev_path))
        dev_loader = DataLoader(dev_dataset, batch_size = 32, shuffle = False,\
                                drop_last = False, num_workers = 4, pin_memory = True)



        train_protocol = DATA_PATH / "ASVspoof5.train.tsv"
        train_path = DATA_PATH / "flac_T"

        d_train, file_list_train = genSpoof_list_asv5(dir_meta = str(train_protocol), is_train = True, is_eval = False)
        

        #d_train == label // 1 : bonafide, 0 : spoof
        train_dataset = TestDataset(list_IDs = file_list_train, base_dir = str(train_path))
        train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = False,\
                                drop_last = False, num_workers = 4, pin_memory = True)


        eval_protocol = DATA_PATH / "ASVspoof5.eval.track_1.tsv"
        eval_path = DATA_PATH / "flac_E_eval"
        d_eval, file_list_eval = genSpoof_list_asv5(dir_meta = str(eval_protocol), is_train = False, is_eval = False)
        

        eval_dataset = TestDataset(list_IDs = file_list_eval[:2000], base_dir = str(eval_path))
        eval_loader = DataLoader(eval_dataset, batch_size = 32, shuffle = False,\
                                drop_last = False, num_workers = 4, pin_memory = True)

        atk_cat, atk_type = None, None


        with torch.no_grad():
            model.eval() #평가 모드로 전환, drop X
            all_embeddings = []
            if data_type == 'train':
                all_labels = [d_train[key] for key in file_list_train]
                atk_cat, atk_type = get_attack_type(dir_meta = str(train_protocol), is_train = True, is_eval = False)

                for batch in tqdm(train_loader, desc = "Extracting Train Embeddings..."):
                    inputs, utt_id = batch
                    inputs = inputs.to(device)
                    embeddings, output = model(inputs)
                    all_embeddings.append(embeddings.cpu().numpy())

            elif data_type == 'dev':
                all_labels = [d_dev[key] for key in file_list_dev[:2000]]
                atk_cat, atk_type = get_attack_type(dir_meta = str(dev_protocol), is_train = False, is_eval = False)
                atk_cat = atk_cat[:2000]
                atk_type = atk_type[:2000]

                for batch in tqdm(dev_loader, desc = "Extracting Dev Embeddings..."):
                    inputs, utt_id = batch
                    inputs = inputs.to(device)
                    embeddings, output = model(inputs)
                    all_embeddings.append(embeddings.cpu().numpy())

            elif data_type == 'eval':
                all_labels = [d_eval[key] for key in file_list_eval[:2000]]
                atk_cat, atk_type = get_attack_type(dir_meta = str(eval_protocol), is_train = False, is_eval = True)
                atk_cat = atk_cat[:2000]
                atk_type = atk_type[:2000]

                for batch in tqdm(eval_loader, desc = "Extracting Eval Embeddings..."):
                    inputs, utt_id = batch
                    inputs = inputs.to(device)
                    embeddings, output = model(inputs)
                    all_embeddings.append(embeddings.cpu().numpy())
            

                    
        all_embeddings = np.concatenate(all_embeddings, axis=0)

        return all_embeddings, all_labels, atk_cat, atk_type

def visualize_tsne(embeddings, labels, atk_category, atk_type, data_type="unknown"):
    """
    t-SNE 결과를 Seaborn으로 시각화하는 함수.
    'bonafide' 클래스의 색상을 파란색으로 고정하고, Category와 Type별 시각화를 모두 생성합니다.
    """
    print(f"\nRunning t-SNE for {data_type.upper()} set...")
    # 데이터가 너무 많을 경우 t-SNE 계산이 매우 오래 걸릴 수 있으므로, 5000개로 샘플링합니다.
    # 전체 데이터를 사용하려면 이 부분을 주석 처리하세요.
    if len(embeddings) > 5000:
        print(f"Dataset too large ({len(embeddings)}). Sampling 5000 points for visualization.")
        indices = np.random.choice(len(embeddings), 5000, replace=False)
        embeddings = embeddings[indices]
        # labels, atk_category, atk_type도 동일한 인덱스로 샘플링해야 합니다.
        labels = [labels[i] for i in indices]
        atk_category = [atk_category[i] for i in indices]
        atk_type = [atk_type[i] for i in indices]

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    # --- 시각화할 데이터 목록 ---
    # 'Attack Type'과 'Attack Category' 두 종류의 플롯을 생성합니다.
    hue_options = {
        "type": atk_type,
        "category": atk_category
    }

    for plot_type, hue_data in hue_options.items():
        # 데이터프레임 생성
        df = pd.DataFrame()
        df['tsne-2d-one'] = tsne_results[:, 0]
        df['tsne-2d-two'] = tsne_results[:, 1]
        
        # bonafide 샘플의 레이블이 None일 수 있으므로 'bonafide' 문자열로 통일
        df['label'] = ['bonafide' if x is None or pd.isna(x) else x for x in hue_data]
        
        # --- ✅ 커스텀 색상 팔레트 생성 ---
        unique_labels = sorted(df['label'].unique())
        
        # 1. 'bonafide'는 파란색으로 고정
        color_map = {'bonafide': '#1f77b4'}  # Matplotlib 기본 파란색
        
        # 2. 나머지 spoof 공격 유형들에 대해 다른 색상 할당
        spoof_labels = [label for label in unique_labels if label != 'bonafide']
        # gist_rainbow, tab20, hsv 등 다양한 팔레트 사용 가능
        spoof_palette = sns.color_palette("gist_rainbow", len(spoof_labels)) 
        
        # 3. spoof 레이블과 색상을 딕셔너리에 추가
        for label, color in zip(spoof_labels, spoof_palette):
            color_map[label] = color
        # ---------------------------------

        # 시각화
        plt.figure(figsize=(12, 10))
        
        # 범례에 표시할 라벨 개수가 너무 많으면 복잡해지므로 처리
        legend_display = "full"
        if len(unique_labels) > 25:
            print(f"Warning: Too many labels ({len(unique_labels)}) for plot_type '{plot_type}'. Hiding legend.")
            legend_display = False

        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="label",
            hue_order=unique_labels,  # 범례 순서 고정
            palette=color_map,        # 직접 만든 커스텀 팔레트 사용
            data=df,
            legend=legend_display,
            alpha=0.8,
            s=20  # 점 크기 조절
        )
        
        title = f"t-SNE of {data_type.upper()} Set by Attack {plot_type.capitalize()}"
        plt.title(title, fontsize=16)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.tight_layout()
        
        # 동적 파일명으로 저장 (덮어쓰기 방지)
        output_dir = "./vis/"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"tsne_{data_type}_by_{plot_type}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150)
        print(f"t-SNE plot saved to {output_path}")
        plt.close() # 메모리 관리를 위해 plot을 닫아줍니다.

if __name__ == "__main__":
    """CUDA_VISIBLE_DEVICE=3 python visualize_embed.py --data_type"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="dev", choices=["dev", "train", "eval"], help="which data to visualize")
    args = parser.parse_args()
    main(args.data_type)
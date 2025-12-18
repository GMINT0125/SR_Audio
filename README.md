# AASIST 

본 코드는 AASIST 모델을 훈련 및 데이터셋에 대한 평가를 위해 제작되었습니다.

1. ASVspoof 5 Dataset Train
2. ASVspoof 5 + ASVspoof 2019 Dataset Train
3. Evaluation Dataset

## Directory Setting
본 코드는 아래의 Directory setting을 바탕으로 구성되어 있습니다.
결과 재현 시, 데이터 경로를 본인의 데이터셋 경로에 맞게 변경해주셔야 합니다.
세부 내용은 아래의 내용을 참조하시기 바랍니다.

```bash
root
├── app
│   ├── asvspoof5 
│        ├── Baseline-AASIST
│               ├── main.py
│               ├── evaluation_all.py
│               ├── finetune.py
├── data
│   ├── ASVspoof
│   │    ├── ASVspoof5
│   │    ├── ASVspoof2019
│   ├── DFADD
│   │     ├── DFADD_ZIP
│   │         ├── DATASET_tts
│   │         ├── ...
│   ├── UnseenTTS
│   │     ├── cloning
│   │         ├─- ...
│   │         ├── 
│   │     ├── tts
│             ├── tts model foler
│             ├── ...

```
## 0. 환경 설정

```bash
pip install -r requirements.txt
cd Baseline-AASIST
```


## 1.  ASVspoof 5 Dataset Train

### 1.1 학습 데이터 경로 설정
- ```config/AASIST_ASVspoof5.conf``` 내부 ```dataset_path``` 항목 경로에 맞게 변경

### 1.2 실행
- ```bash train.sh 0``` 
    - 학습 결과는 ```./exp_result/AASIST_ASVspoof5_ep{epoch}_bs{batch size}/``` 폴더 내부에 저장됩니다

평가의 경우
- ```bash eval.sh 0``` 실행 (학습과 같은 경로에 결과 저장)



## 2.  ASVspoof 5 + ASVspoof2019 Dataset Train

### 2.1 학습 데이터 경로 설정
- ```finetune.py``` 의 ```main``` 함수 내부 ```python DATA_PATH```변수 데이터셋의 경로에 맞게 변경

### 2.2 실행
- ```bash finetue.sh 0``` 실행
    - 학습 결과는 ```./exp_result/Finetune/```폴더 내부에 저장됩니다.




## 3.  Evaluation Dataset

### 3.1 평가 데이터 경로 및 가중치 경로 설정
- ```evaluation_all.py``` 의 ```main``` 함수 내부 ```model_path```변수를 평가하고자 하는 모델 가중치 경로로 변경
- 코드 내부 ```eval_{dataset}``` 함수 내부의 ```DATA_PATH``` 경로를 데이터셋 경로에 맞게 변경

### 3.2 실행
- ```bash eval_all.sh``` 실행
    - 평가 결과는 ```./exp_result/{Dataset name}/``` 폴더 내부에 저장됩니다.



## 요구사항
- Python: 3.8+ 권장
- CUDA GPU: 권장 (CPU로도 가능하나 매우 느립니다)
- PyTorch: aasist/requirements.txt에 따라 자동 설치됨
- ASVspoof5, ASVspoof2019, DFADD, UnseenTTS 데이터 경로에 맞게 설치 필요

## 재현 EER
|Train data|ASV5|ASV2019|D1|D2|D3|F1|F2|Unseen(종합)|
|------|------|------|------|------|------|------|------|------|
|ASV5|27.14|-|4.80|3.32|0.29|1.17|8.03|28.51
|ASV2019|-|0.83|37.67|27.33|34.83|37.33|49.33|17.79|
|ASV5+2019|25.64|2.01|18.15|23.68|29.81|40.66|26.86|
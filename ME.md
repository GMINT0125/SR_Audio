# AASIST 

본 코드는 AASIST 모델을 훈련 및 데이터셋에 대한 평가를 위해 제작되었습니다.

1. ASVspoof 5 Dataset Train
2. ASVspoof 5 + ASVspoof2019 Dataset Train
3. Evaluation Dataset

## Directory Setting


## 1.  ASVspoof 5 Dataset Train

### 1.1 학습 데이터 경로 설정
- ```config/AASIST_ASVspoof5.conf``` 내부 dataset_path 항목 경로에 맞게 변경

### 1.2 실행
- ```bash train.sh 0``` 
- 실행 학습 결과는 ```exp_result/AASIST_ASVspoof5_ep{epoch}_bs{batch size}``` 형태로 저장됩니다

평가의 경우
- ```bash eval.sh 0``` 실행 (학습과 같은 경로에 결과 저장)

## 2.  ASVspoof 5 + ASVspoof2019 Dataset Train

### 2.1 학습 데이터 경로 설정

### 2.2 실행
- ```bash finetue.sh``` 실행
- 학습 결과는 ```exp_result/Finetune```에 저장됩니다.

## 3.  Evaluation Dataset

### 3.1 평가 데이터 경로 설정

### 3.2 실행
- ```bash eval_all.sh``` 실행
- 평가 결과는 ```exp_result/{Dataset name}``` 형태로 저장됩니다.
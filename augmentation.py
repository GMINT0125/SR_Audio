"""
Offline audio codec 적용
- 사용 라이브러리: ffmpeg-python
- 적용 코덱: Opus, MP3, AAC
- ASVspoof5 경로에 맞게 수정됨
"""

import ffmpeg  # ffmpeg-python 라이브러리
from pathlib import Path
import os
import random 
import shutil
from tqdm import tqdm  


# --- 코덱 프로필 정의 (안전한 리스트) ---
PROFILES_16K_STATIC = [
    {"format": "ogg", "acodec": "libopus", "audio_bitrate": "8k"},
    {"format": "ogg", "acodec": "libopus", "audio_bitrate": "16k"},
    {"format": "mp3", "acodec": "libmp3lame", "audio_bitrate": "32k"},
    {"format": "mp4", "acodec": "aac", "audio_bitrate": "32k"},
]
PROFILES_8K_STATIC = [
    {"format": "ogg", "acodec": "libopus", "audio_bitrate": "6k"},
    {"format": "ogg", "acodec": "libopus", "audio_bitrate": "12k"},
]

# --- 3. ffmpeg-python 처리 함수 (수정 없음) ---

def process_file_with_ffmpeg_python(input_file, output_file, profile, sr_type):
    """
    ffmpeg-python을 사용해 (인코딩 -> 파이프 -> 디코딩) 시뮬레이션
    """
    
    # --- Part 1: 인코더 설정 (Input -> Encode) ---
    input_stream = ffmpeg.input(str(input_file))
    print("sr_type:", sr_type)
    if sr_type == '8k':
        input_stream = ffmpeg.filter(input_stream, 'aresample', 8000)
    else:
        input_stream = ffmpeg.filter(input_stream, 'aresample', 16000)

    # --- Part 2: 디코더 설정 (Decode -> Output) ---
    encoder_process = (
        input_stream
        .output('pipe:', **profile)
        .run_async(pipe_stdout=True, quiet=False)
    )

    decoder_process = (
        ffmpeg
        .input('pipe:', format=profile['format'])
        .output(str(output_file), acodec='flac', ar=16000)
        .run_async(pipe_stdin=encoder_process.stdout, quiet=True)
    )
    
    decoder_process.wait()
    encoder_process.wait()


def main():

    # (수정) pathlib.Path 객체로 경로 관리 통일
    ROOT_DATA_PATH = Path("../../../data/ASVspoof/ASVspoof5") # (수정) 따옴표 오류 수정
    TRAIN_DATA_PATH = ROOT_DATA_PATH / "flac_T"
    OUTPUT_PATH = ROOT_DATA_PATH / "codec_flac_T" # (수정) 변수 이름
    
    # (수정) pathlib 방식으로 디렉터리 생성
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True) 

    tsv_protocol = ROOT_DATA_PATH / "ASVspoof5.train.tsv" 
    with open(tsv_protocol, "r") as f:
        lines = f.readlines()
    
    file_list = []
    for line in lines[1:]: # 헤더(첫 줄)가 있다면 [1:]로 스킵
        # ASVspoof5.train.tsv 포맷 (공백 기준):
        # SUBSET FILENAME - - - - - - LABEL -
        parts = line.strip().split(" ")
        key = parts[1]   # FILENAME (예: T_0000000)
        file_list.append(key)
    
    all_profiles_with_sr = [("16k", p) for p in PROFILES_16K_STATIC] + \
                           [("8k", p) for p in PROFILES_8K_STATIC]
    
    num_clean_copies = 2 
    all_profiles_with_sr.extend([("16k", "C00_CLEAN")] * num_clean_copies)
    
    print(f"총 {len(file_list)}개의 파일 처리 시작...")

    for key in tqdm(file_list, ncols=80):
        input_path = TRAIN_DATA_PATH / f"{key}.flac"
        output_path = OUTPUT_PATH / f"{key}.flac"
        
        if not input_path.exists():
            print(f"[경고] 원본 파일 없음: {input_path}")
            continue
            
        sr_type, profile = random.choice(all_profiles_with_sr)
        
        try:
            if profile == "C00_CLEAN":
                shutil.copyfile(input_path, output_path)
                continue
                
            process_file_with_ffmpeg_python(
                input_path, output_path, profile, sr_type
            )
            
        except ffmpeg.Error as e:
            print(f"[에러] {key} 처리 중 오류 발생:")
            print(e.stderr.decode('utf8')) 
            shutil.copyfile(input_path, output_path)
        except Exception as e:
            print(f"[일반 에러] {key}: {e}")
            shutil.copyfile(input_path, output_path)

    print("오프라인 증강 완료!")
    print(f"새 데이터셋이 {OUTPUT_PATH}에 저장되었습니다.") # (수정) OUTPUT_DIR -> OUTPUT_PATH


if __name__ == "__main__":
    main()
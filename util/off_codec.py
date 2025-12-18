"""
ASVSpoof5 데이터셋의 오디오 파일에 무작위로 오디오 코덱을 적용하여 증강하는 스크립트입니다.
"""

import os
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from DAREA.src.darea.augmentation.codecs import CodecAugmentation

SRC_DIR = "../../../data/ASVspoof/ASVspoof5/flac_T/"
DST_DIR = "../../../data/ASVspoof/ASVspoof5/flac_T_codec/"
os.makedirs(DST_DIR, exist_ok=True)

CODEC_LIST = [
    ("mp3", 32000),
    ("ogg-opus", 32000),
    ("ogg-vorbis", 32000),
]

SAMPLE_RATE = 16000


def process_file(file):
    src_path = os.path.join(SRC_DIR, file)
    dst_path = os.path.join(DST_DIR, file)

    if os.path.exists(dst_path):
        return

    try:
        x, sr = sf.read(src_path)
        if len(x.shape) > 1:
            x = x[:, 0]
        if sr != SAMPLE_RATE:
            print(f"[WARN] {file} sample rate mismatch ({sr})")

        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 무작위 codec 선택
        codec_fmt, bitrate = CODEC_LIST[np.random.randint(len(CODEC_LIST))]

        try:
            codec = CodecAugmentation(
                format=codec_fmt,
                sample_rate=SAMPLE_RATE,
                bitrate=bitrate,
                q_factor=None
            )
            with torch.no_grad():
                x_codec = codec(x_t).squeeze().numpy()

            # clipping 방지
            x_codec = np.clip(x_codec, -1.0, 1.0)
            sf.write(dst_path, x_codec, SAMPLE_RATE)

        except Exception as e:
            print(f"[ERROR] {file} codec={codec_fmt} failed: {e}")

    except Exception as e:
        print(f"[SKIP] Failed to process {file}: {e}")


if __name__ == "__main__":
    files = [f for f in os.listdir(SRC_DIR) if f.endswith(".flac")]
    print(f"Total files to augment: {len(files)}")

    with Pool(processes=min(8, cpu_count())) as pool:
        for _ in tqdm(pool.imap_unordered(process_file, files), total=len(files)):
            pass

    print("Augmentation complete. Saved to:", DST_DIR)

import librosa
import numpy as np
import os
import list_file_class
from tqdm import tqdm

n_fft = 4096
n_hop = 1024
n_mfcc = 40

li = list_file_class.li # 클래스, 인스턴스 분류

m = 0
n = 0

for Class in li:
    for file_name in os.listdir('./Data/' + Class):
        audio, sample_rate = librosa.load('./Data/' + Class + '/' + file_name, res_type='kaiser_fast', sr=44100)
        # duration 설정 값에 따라 초단위로 오디오 파일 범위 추출
        #mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=n_hop, n_fft=n_fft)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_fft=4096, hop_length=n_hop, n_mfcc=n_mfcc)
        mfccs = np.array(mfccs)
        print(mfccs.shape[1])
        m = max(m, mfccs.shape[1])
        n = n + 1

print(m)

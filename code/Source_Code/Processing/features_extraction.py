import librosa
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import cv2

# 고정값
max_pad_len = 1292
max_pad_height = 40
n_hop = 1024

# 변경값
t = 'test'

# librosa 하이퍼파라미터
libro_set = {
    'low':{'n_fft':1024, 'f_len':1024, 'n_mfcc':10, 'n_mels':10},
    'mid':{'n_fft':2048, 'f_len':2048, 'n_mfcc':20, 'n_mels':20},
    'high':{'n_fft':4096, 'f_len':4096, 'n_mfcc':40, 'n_mels':40},
    'test':{'n_fft':4096, 'f_len':4096, 'n_mfcc':40, 'n_mels':40}
}

class extract:
    audio = 0
    sample_rate = 0
    
    data=[[],[],[],[],[],[]]
    
    def set_audio(self, audio):
        self.audio = audio
        
    def Set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate
    
    def features(self, class_label):
        f = []
        f.append(librosa.feature.mfcc(y=self.audio, sr=self.sample_rate, hop_length=n_hop, n_fft=libro_set[t]['n_fft'], n_mfcc=libro_set[t]['n_mfcc']))
        f.append(librosa.feature.zero_crossing_rate(y=self.audio, hop_length=n_hop, frame_length=libro_set[t]['f_len']))
        f.append(librosa.feature.spectral_centroid(y=self.audio, sr=self.sample_rate, hop_length=n_hop, n_fft=libro_set[t]['n_fft']))
        f.append(librosa.feature.spectral_rolloff(y=self.audio, sr=self.sample_rate, hop_length=n_hop, n_fft=libro_set[t]['n_fft']))
        f.append(librosa.feature.melspectrogram(y=self.audio, sr=self.sample_rate, hop_length=n_hop, n_fft=libro_set[t]['n_fft'], n_mels=libro_set[t]['n_mels']))
        f.append(librosa.feature.chroma_stft(y=self.audio, sr=self.sample_rate, hop_length=n_hop, n_fft=libro_set[t]['n_fft']))
        
        for i in range(6):
            self.pad(i, f[i], class_label)
        
    def pad(self, num, feat, class_label):
        if num in (1, 2, 3): 
            # zero_padding(width, height all)
            pad_width = max_pad_len - feat.shape[1]
            pad_height = max_pad_height - feat.shape[0]
            feat = np.pad(feat, pad_width=((0, pad_height), (0, pad_width)), mode="constant")
        else:
            # resize 보간(width)
            pad_width = max_pad_len - feat.shape[1]
            feat = np.pad(feat, pad_width=((0, 0), (0, pad_width)), mode="constant")
            feat = cv2.resize(feat, (max_pad_len, max_pad_height), interpolation=cv2.INTER_CUBIC)
        
        self.data[num].append([feat, class_label])
            
    def save(self, num, tag):
        features_df = pd.DataFrame(self.data[num], columns=['feature', 'class_label']) 
        features_df.to_json('./Processed_Data/data_' + tag + '_' + t + '.json')
        

if __name__ == '__main__':
    li = ['mfcc', 'zcr', 'centroid', 'rolloff', 'mel', 'chroma']
    tmp = extract()
    
    # 각종 경로 설정
    metadata = pd.read_csv('./Meta_data/Data_t.csv')

    if not os.path.exists("./Processed Data"):
        os.mkdir("./Processed Data")
    
    # 각 사운드 파일별 특징 추출
    n = 0
    for index, row in tqdm(metadata.iterrows()):
        class_label = row["label"]
        file_name = './Data/' + str(row["class"]) + '/' + str(row["filename"])
        # print(file_name)
        try:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', sr=44100)
            tmp.set_audio(audio)
            tmp.Set_sample_rate(sample_rate)
            tmp.features(class_label)
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
        
        n += 1
        # print(n, "개 추출 완료")
        
    for i in range(6):
        tmp.save(i, li[i])
    
    del tmp    
        
    print(f'{n}개 피처 추출 완료')
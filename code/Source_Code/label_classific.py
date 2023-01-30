from tensorflow.keras.models import load_model
import librosa
import numpy as np
import librosa.display
from Processing.list_file_class import Class_ins
from pydub import AudioSegment
from pickle import load

num_rows, num_columns = 40, 1292
max_pad_len = 1292
n_fft = 4096
n_hop = 1024
n_mfcc = 40


def Result(filename):
    
    m4a_file = filename
    wav_filename = filename.split('.')[0] + '.wav'
    
    track = AudioSegment.from_file(m4a_file, format='m4a')
    file_handle = track.export(wav_filename, format='wav')
    
    # 1. 실무에 사용할 데이터 준비하기
    audio, sample_rate = librosa.load(file_handle, res_type='kaiser_fast', sr=44100, duration=30)
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=n_hop, n_fft=n_fft)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode="constant")

    mfccs = np.array(mfccs)
    
    # 2. 모델 불러오기
    model = load_model('./Saved_Model/Weight_best.hdf5')
    scaler = load(open('./Saved_Scale/minmax_scaler.pkl', 'rb'))
    
    x_test = scaler.transform(mfccs.reshape(1, num_rows * num_columns))

    predicted_classes = np.argmax(model.predict(x_test), axis=1)
    
    return Class_ins(int(predicted_classes))
import soundfile
import matplotlib.pyplot as plt
import librosa.display

# 원음
audio1, sample_rate1 = librosa.load('./Data1/2.5톤/S-211129_V_101_D_002_1.wav', res_type='kaiser_fast', sr=44100)
# 합성음
audio2, sample_rate2 = librosa.load('./Data1/2.5톤/S-211129_V_101_D_002_2.wav', res_type='kaiser_fast', sr=44100)

audio_len = 220500 # 오디오 길이
sample_rate = 44100

k = 0.4  # 합성하고자 하는 소리 비율
audio_sum = []

for i in range(audio_len):
    audio_sum.append((1 - k) * audio1[i] + k * audio2[i]) # 오디오 합치는 부분

soundfile.write('./result_sum2.wav', audio_sum, sample_rate, format='wav') # 합성 파일 저장
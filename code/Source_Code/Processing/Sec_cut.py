import os
import soundfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display

for d in tqdm(os.listdir('./Data')):
    for f in tqdm(os.listdir('./Data/' + d)):
        audio, sample_rate = librosa.load('./Data/' + d + '/' + f, res_type='kaiser_fast', sr=44100)
        duration = librosa.get_duration(audio, sample_rate)
        j = 0
        z = 1
        for i in tqdm(range(5, 31, 5)):
            try:
                if j <= int(duration):
                    audio_split = audio[sample_rate * j:sample_rate * i]
                    soundfile.write('./Data_Split/' + d + '/' + f.split('.')[0] + '_' + str(z) + '.wav', audio_split, sample_rate, format='wav')
                    plt.ylim(-1, 1)
                    librosa.display.waveshow(y=audio_split, sr=sample_rate)
                    plt.savefig('./Data_Split_img/' + d + '/' + f.split('.')[0] + '_' + str(z) + '.png')
            except:
                print(f.split('.')[0] + '_' + str(z) + '변환실패')
            finally:
                j = i
                z += 1
                plt.clf()
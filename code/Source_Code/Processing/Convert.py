from pydub import AudioSegment
import os
import list_file_class

for i in list_file_class.li:
    print(i)
    n = 0
    err = 0
    for j in os.listdir('./Data/' + i): # 해당 폴더 안에 파일명 리스트화
        try:
            if os.path.isdir('./Data/' + i)==True and os.path.isfile('./Data/' + i + '/' + j.split('.')[0] + '.mp3')==True:
                audSeg = AudioSegment.from_mp3('./Data/' + i + '/' + j) # mp3 원본 파일 로드
                audSeg.export('./Data/' + i + '/' + j.split('.')[0] + '.wav', format="wav") # wav 파일로 변환
                n += 1
                print('{}개 변환'.format(n))
                os.remove('./Data/' + i + '/' + j.split('.')[0] + '.mp3') # wav 파일로 변환 후 mp3 파일 삭제
            else:
                print(j + '이미 변환 되어 있는 파일입니다.')
        except:
            err += 1
            print('./Data/' + i + '/' + j + '등 {}개 변환 에러'.format(err))
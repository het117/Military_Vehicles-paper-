import os
import pandas as pd
import list_file_class

li = list_file_class.li # 클래스, 인스턴스 분류

f, c, l = [], [], []
n = 0

for i in li:
    for j in os.listdir('./Data/' + i):
        f.append(j)
        c.append(i)
        l.append(n)
    n += 1
    print('{}개 추출'.format(n))

df = pd.DataFrame(f, columns=['filename']) # 파일명 입력
df['class'] = c # 클래스 입력
df['label'] = l # 라벨링
df.to_csv("./Meta_data/Data.csv", index=False) # csv 파일 생성
import numpy as np
import pandas as pd
from Param import data_init
import NN_model as nn
import matplotlib.pyplot as plt
import seaborn as sns

import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Processing import list_file_class

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix

import warnings
warnings.filterwarnings('ignore')

MODEL_SAVE = "./Saved Model"
num_rows, num_columns, num_channels = 40, 1292, 1

scaler = MinMaxScaler()

model_list = {
    # 'New_Model':nn.New_Model,
    # 'VGG':nn.VGG,
    # 'Logistic':nn.Logistic,
    'SVM':nn.SVM,
    # 'LeNet':nn.LeNet,
    # 'ResNet':nn.ResNet
}


def d_learning(tag, scale):
    features_df = pd.read_json('./Processed_Data/data_' + tag + '_' + scale + '.json')

    x = np.array(features_df.feature.tolist())
    y = np.array(features_df.class_label.tolist())

    # x = x.reshape(x.shape[0], num_rows, num_columns)
    # x = np.expand_dims(x, -1)

    le = LabelEncoder()
    y = to_categorical(le.fit_transform(y))

    data_in_d = train_test_s(x, y, 'd')
    data_in_m = train_test_s(x, y, 'm')
    
    for key, value in model_list.items():
        if key in ('LeNet', 'VGG', 'ResNet'):
            train_time = value(data_in_d)
            model = load_model("./Saved Model/Weight_best.hdf5")
            result(model, data_in_d, train_time, tag, scale, key)
        elif key in ('SVM', 'Logistic'):
            train_time = value(data_in_m)
            model = load_model("./Saved Model/Weight_best.hdf5")
            result(model, data_in_m, train_time, tag, scale, key)


def train_test_s(x, y, check):
    data_in = data_init()
    
    if check == 'd':
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)
        
        x_train = scaler.fit_transform(x_train.reshape(x_train.shape[0], num_rows * num_columns))
        x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns)
        x_train = np.expand_dims(x_train, -1)
        
        x_val = scaler.transform(x_val.reshape(x_val.shape[0], num_rows * num_columns))
        x_val = x_val.reshape(x_val.shape[0], num_rows, num_columns)
        x_val = np.expand_dims(x_val, -1)
        
        x_test = scaler.transform(x_test.reshape(x_test.shape[0], num_rows * num_columns))
        x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns)
        x_test = np.expand_dims(x_test, -1)
        
        print(x_train.shape, y_train.shape)
        print(x_val.shape, y_val.shape)
        print(x_test.shape, y_test.shape)
        
        data_in.d_set_total(x_train, x_test, y_train, y_test, x_val, y_val)
    
    if check == 'm':
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)
        
        x_train = x_train.reshape(x_train.shape[0], num_rows * num_columns)
        x_train = scaler.fit_transform(x_train)
        
        x_val = x_val.reshape(x_val.shape[0], num_rows * num_columns)
        x_val = scaler.transform(x_val)
        
        x_test = x_test.reshape(x_test.shape[0], num_rows * num_columns)
        x_test = scaler.transform(x_test)
        
        print(x_train.shape, y_train.shape)
        print(x_val.shape, y_val.shape)
        print(x_test.shape, y_test.shape)
        
        data_in.d_set_total(x_train, x_test, y_train, y_test, x_val, y_val)
    
    return data_in


def result(model, data_in, train_time, tag, scale, a):
        data = {'t_score':[], 'v_score':[], 'accuracy':[], 'precision':[], 'recall':[], 'f1':[], 'train_time':[]}
        
        data['t_score'].append(round(model.evaluate(data_in.get_x_train(), data_in.get_y_train(), verbose=0)[1] * 100, 2))
        data['v_score'].append(round(model.evaluate(data_in.get_x_val(), data_in.get_y_val(), verbose=0)[1] * 100, 2))
        data['accuracy'].append(round(model.evaluate(data_in.get_x_test(), data_in.get_y_test(), verbose=0)[1] * 100, 2))

        result = model.predict(data_in.get_x_test())
        
        y_pred = np.array([np.argmax(result[i]) for i in range(result.shape[0])])
        y_test = np.argmax(data_in.get_y_test(), axis=1)
        
        data['precision'].append(round(precision_score(y_test, y_pred, average='weighted') * 100, 2))
        data['recall'].append(round(recall_score(y_test, y_pred, average='weighted') * 100, 2))
        data['f1'].append(round(f1_score(y_test, y_pred, average='weighted') * 100, 2))
        data['train_time'].append(round(train_time, 2))
        
        df = pd.DataFrame(data)
        df.to_csv('./L_Result/d_result_' + tag + '_' + scale + '_' + a + '.csv', index=False)  # csv 파일 생성

        print(df)
        
        # plot_confusion(y_test, y_pred)


def plot_confusion(y_test, y_preds):
    cm = confusion_matrix(y_test, y_preds)

    plt.figure(figsize=(16,9))
    sns.heatmap(
        cm,
        annot=True,
        
        xticklabels=['2.5톤', '5톤', '9.5톤', '10톤', '27톤',
      'K-1', 'K1a1', 'k10탄약운반차', 'K56', 'K77',
      'K288a1', 'K200', 'K800', 'km9ace', '교량전차',
      '다목적굴착기', '대형버스', '부식차', '살수차', '승용차',
      '장애물개척자', '통신가설차량', '화생방정찰차'],
        
        yticklabels=['2.5톤', '5톤', '9.5톤', '10톤', '27톤',
      'K-1', 'K1a1', 'k10탄약운반차', 'K56', 'K77',
      'K288a1', 'K200', 'K800', 'km9ace', '교량전차',
      '다목적굴착기', '대형버스', '부식차', '살수차', '승용차',
      '장애물개척자', '통신가설차량', '화생방정찰차']
    )
    plt.show()
    

d_learning('mfcc', 'test')
# d_learning('zcr', 'test')
# d_learning('centroid', 'test')
# d_learning('rolloff', 'test')
# d_learning('mel', 'test')
# d_learning('chroma', 'test')
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from Param import data_init, LR_param_grid, RF_param_grid, SVM_param_grid, Tree_param_grid

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


def m_learning(tag, scale, num_columns=1292):  # 216 or 1292
    features_df = pd.read_json('./Processed_Data/data_' + tag + '_' + scale + '.json')
    
    data_in = data_init()

    x = np.array(features_df.feature.tolist())
    y = np.array(features_df.class_label.tolist())

    num_rows = x.shape[1]
    x = x.reshape(x.shape[0], num_rows * num_columns)
    
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    
    data_in.m_set_total(x_train, x_test, y_train, y_test)
    
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    print(model.score(x_train, y_train))
    print(accuracy_score(y_test, y_pred))
    
    
    # LR_param = LR_param_grid; RF_param = RF_param_grid; SVM_param = SVM_param_grid; Tree_param = Tree_param_grid
    
    # algo(list(ParameterGrid(LR_param)), LR_param, tag, scale, 'LR', data_in)
    # algo(list(ParameterGrid(RF_param)), RF_param, tag, scale, 'RF', data_in)
    # algo(list(ParameterGrid(SVM_param)), SVM_param, tag, scale, 'SVM', data_in)
    # algo(list(ParameterGrid(Tree_param)), Tree_param, tag, scale, 'Tree', data_in)
    
    
def algo(t, param_grid, tag, scale, a, data_in):
    data = {'t_score':[], 'v_score':[], 'accuracy':[], 'precision':[], 'recall':[], 'f1':[], 'train_time':[]}

    for key in param_grid.keys():data[key] = []
   
    skfold = StratifiedKFold(n_splits=4)
    
    t_feature = data_in.get_x_train(); t_label = data_in.get_y_train()
        
    for param in tqdm(t):
        if a == 'LR':classifier = LogisticRegression(**param)
        if a == 'RF':classifier = RandomForestClassifier(**param)
        if a == 'SVM':classifier = svm.SVC(**param)
        if a == 'Tree':classifier = DecisionTreeClassifier(**param)
        
        t_accuracy = []
        cv_accuracy = []
        e_accuracy = []
        
        for train_index, test_index in tqdm(skfold.split(t_feature, t_label)):
            x_train, x_val = t_feature[train_index], t_feature[test_index]
            y_train, y_val = t_label[train_index], t_label[test_index]
            
            start = time.time()
            classifier.fit(x_train, y_train)
            train_time = time.time() - start
            
            t_accuracy.append(classifier.score(x_train, y_train))
            y_pred = classifier.predict(x_val)
            cv_accuracy.append(accuracy_score(y_val, y_pred))
            y_pred = classifier.predict(data_in.get_x_test())
            e_accuracy.append(accuracy_score(data_in.get_y_test(), y_pred))
        
        for key in data.keys():
            if key in('t_score, v_score, accuracy, precision, recall, f1, train_time'):
                if key == 't_score':data['t_score'].append(round(np.mean(t_accuracy) * 100, 2))
                if key == 'v_score':data['v_score'].append(round(np.mean(cv_accuracy) * 100, 2))
                if key == 'accuracy':data['accuracy'].append(round(np.mean(e_accuracy) * 100, 2))
                if key == 'precision':data['precision'].append(round(precision_score(data_in.get_y_test(), y_pred, average='weighted') * 100, 2))
                if key == 'recall':data['recall'].append(round(recall_score(data_in.get_y_test(), y_pred, average='weighted') * 100, 2))
                if key == 'f1':data['f1'].append(round(f1_score(data_in.get_y_test(), y_pred, average='weighted') * 100, 2))
                if key == 'train_time':data['train_time'].append(round(train_time, 2))
            else:
                data[key].append(param[key])

    df = pd.DataFrame(data)
    df.to_csv('./m_Result/m_result_' + tag + '_' + scale + '_' + a + '.csv', index=False)  # csv 파일 생성
    
    print(df)
    

if __name__ == '__main__':
    m_learning('mfcc', 'test_test1')
    # m_learning('mfcc', 'high')
    # m_learning('zcr', 'high')
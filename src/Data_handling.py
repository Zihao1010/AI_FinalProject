import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

class DataHandling:
    def __init__(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train

    def SMOT(self,r):
        print('Data Amount Before Balance:',self.y_train[self.y_train==1].shape,self.y_train[self.y_train==0].shape)
        smote=SMOTE(sampling_strategy=r,random_state=2)
        X_train_bal,y_train_bal=smote.fit_resample(self.X_train,self.y_train.ravel())
        print('Data Amount After Balance:',y_train_bal[y_train_bal == 1].shape, y_train_bal[y_train_bal == 0].shape)
        return X_train_bal,y_train_bal

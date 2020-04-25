import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class DataCleaning:
    def __init__(self):
        self.oridata=pd.read_csv('data/train.csv')

    def Data_Cleaning(self):
        self.oridata=self.oridata.replace(-1,np.nan)
        null_list=[]
        #Print the null value percent of every column, filter these columns with high percent
        for i in self.oridata.columns:
            null_count=self.oridata[i].isnull().sum()
            if null_count>0:
                null_list.append((i,null_count))
        # for i in range(len(null_list)):
        #     print('column:',null_list[i][0],', nan percent: {:.4%}'.format(null_list[i][1]/self.oridata.shape[0]))
        #Drop column 'ps_car_03_cat' and 'ps_car_05_cat'
        self.oridata=self.oridata.drop(['ps_car_03_cat','ps_car_05_cat'],axis=1)
        #Fill na with mean value for the remain colomns which value type is number
        self.oridata.ps_reg_03[self.oridata.ps_reg_03.isnull()]=self.oridata.ps_reg_03.dropna().mean()
        self.oridata.ps_car_11[self.oridata.ps_car_11.isnull()]=self.oridata.ps_car_11.dropna().mean()
        self.oridata.ps_car_12[self.oridata.ps_car_12.isnull()]=self.oridata.ps_car_12.dropna().mean()
        self.oridata.ps_car_14[self.oridata.ps_car_14.isnull()]=self.oridata.ps_car_14.dropna().mean()
        #Fill na with mode for the remain columns which value type is catrgories
        self.oridata.ps_ind_02_cat[self.oridata.ps_ind_02_cat.isnull()]=self.oridata.ps_ind_02_cat.dropna().mode()[0]
        self.oridata.ps_ind_04_cat[self.oridata.ps_ind_04_cat.isnull()]=self.oridata.ps_ind_04_cat.dropna().mode()[0]
        self.oridata.ps_ind_05_cat[self.oridata.ps_ind_05_cat.isnull()]=self.oridata.ps_ind_05_cat.dropna().mode()[0]
        self.oridata.ps_car_01_cat[self.oridata.ps_car_01_cat.isnull()]=self.oridata.ps_car_01_cat.dropna().mode()[0]
        self.oridata.ps_car_02_cat[self.oridata.ps_car_02_cat.isnull()]=self.oridata.ps_car_02_cat.dropna().mode()[0]
        self.oridata.ps_car_07_cat[self.oridata.ps_car_07_cat.isnull()]=self.oridata.ps_car_07_cat.dropna().mode()[0]
        self.oridata.ps_car_09_cat[self.oridata.ps_car_09_cat.isnull()]=self.oridata.ps_car_09_cat.dropna().mode()[0]

        # print(self.oridata.info())
        dummy_handle=['ps_ind_02_cat','ps_ind_04_cat','ps_ind_05_cat','ps_car_01_cat','ps_car_02_cat','ps_car_04_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat']

        for i in dummy_handle:
            tmp=pd.get_dummies(self.oridata[i],prefix=i)
            self.oridata=self.oridata.drop([i],axis=1)
            self.oridata=self.oridata.join(tmp)

        self.oridata=self.oridata.drop(['id'],axis=1)





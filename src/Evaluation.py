import numpy as np

class Evaluation:
    def __init__(self):
        self.gini=0

    def gini_score(self,y_true,predict_proba):
        y_true = np.asarray(y_true)
        y_true=y_true[np.argsort(predict_proba)]
        n_true=0
        delt=0
        length=len(y_true)
        for i in range(length-1,-1,-1):
            tmp=y_true[i]
            n_true+=tmp
            self.gini+=tmp*delt
            delt+=1-tmp
        self.gini=1-2*self.gini/(n_true*(length-n_true))
        return self.gini

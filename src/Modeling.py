import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, confusion_matrix
import Data_handling
import Evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import sklearn.metrics



class Modeling:
    def __init__(self,data):
        self.data=data
        #Since we don't have real target in test.csv, so we'll gonna random split the train data to 4:1 to modeling.
        #We use X
        self.X_trainval,self.X_test,self.y_trainval,self.y_test=train_test_split(self.data.drop('target',axis=1),self.data['target'])
        self.gini=0

    def Neural_Network(self):
        # Balance Data
        r = 0.25
        Handle_Data = Data_handling.DataHandling(self.X_trainval, self.y_trainval)
        X_trainval_bal, y_trainval_bal = Handle_Data.SMOT(r)

        # Standarize data
        scaler = StandardScaler().fit(X_trainval_bal)
        X_trainval_bal_transformed = scaler.transform(X_trainval_bal)
        X_test_transformed = scaler.transform(self.X_test)

        # PCA
        Data_pca = PCA(n_components=200).fit(X_trainval_bal_transformed)
        X_trainval_bal_transformed = Data_pca.transform(X_trainval_bal_transformed)
        X_test_transformed = Data_pca.transform(X_test_transformed)

        activations = ['tanh', 'relu']
        alphas = [0.6, 0.7]
        solvers = ['sgd', 'adam']

        Eva = Evaluation.Evaluation()
        best_score = 0

        # Get the best parameter, since this is a long-time work, so after we found the best parameter we need, we comment the following code.

        for acti in activations:
            for solver in solvers:
                MLPmodel = MLPClassifier(solver=solver, activation=acti, random_state=10, hidden_layer_sizes=[100, 10],
                                             alpha=0.5, max_iter=5000)
                scorer = make_scorer(Eva.gini_score, needs_proba=True)
                score = cross_val_score(MLPmodel, X_trainval_bal_transformed, y_trainval_bal, cv=5, scoring=scorer)
                    # print(score)
                score = score.mean()
                print("When parameter activation=",acti,", parameter solver=",solver,":\nMean score is", score)
                if score >= best_score:
                    best_score = score
                    best_acti = acti
                    best_solver = solver

        # Get the best model
        BestMLPModel = MLPClassifier(solver=best_solver, activation=best_acti, random_state=10, hidden_layer_sizes=[100, 10],
                                             alpha=0.5, max_iter=5000).fit(X_trainval_bal_transformed, y_trainval_bal)

        #Here we use the best parameter we found above to train the model
        # BestMLPModel = MLPClassifier(solver='sgd', activation='relu', random_state=10,
        #                              hidden_layer_sizes=[100, 10],
        #                              alpha=0.6, max_iter=5000).fit(X_trainval_bal_transformed, y_trainval_bal)
        predict_proba = BestMLPModel.predict_proba(X_test_transformed)[:, 1]
        self.gini = Eva.gini_score(self.y_test, predict_proba)
        return BestMLPModel, self.gini

    def Logistic_Regression(self):
        #Balance Data
        r=0.25
        Handle_Data=Data_handling.DataHandling(self.X_trainval,self.y_trainval)
        X_trainval_bal,y_trainval_bal=Handle_Data.SMOT(r)

        #Standarize data
        scaler=StandardScaler().fit(X_trainval_bal)
        X_trainval_bal_transformed=scaler.transform(X_trainval_bal)
        X_test_transformed=scaler.transform(self.X_test)

        #Train Model
        penalty=['l1','l2']
        c=[0.01,1.0]
        best_penalty=''
        best_c=0
        best_score=0
        Eva = Evaluation.Evaluation()
        for i in penalty:
            for j in c:
                LogRegModel= LogisticRegression(penalty=i,C=j,solver='liblinear')
                scorer=make_scorer(Eva.gini_score,needs_proba=True)
                score=cross_val_score(LogRegModel,X_trainval_bal_transformed,y_trainval_bal,cv=5,scoring=scorer)
                print(score)
                score=score.mean()
                print("When parameter penalty=",i,", parameter C=",j,":\nMean score is", score)
                if score>=best_score:
                    best_score=score
                    best_penalty=i
                    best_c=j

        BestLogRegModel = LogisticRegression(penalty=best_penalty, solver='liblinear', C=best_c).fit(
            X_trainval_bal_transformed, y_trainval_bal)

        #Get the best model using the best parameter we trained
        # BestLogRegModel=LogisticRegression(penalty='l2',solver='liblinear',C=0.01).fit(X_trainval_bal_transformed,y_trainval_bal)
        predict_proba=BestLogRegModel.predict_proba(X_test_transformed)[:,1]
        self.gini=Eva.gini_score(self.y_test,predict_proba)

        return BestLogRegModel,self.gini

    def RandomForest(self):
        #Balance Data
        r = 0.25
        Handle_Data = Data_handling.DataHandling(self.X_trainval, self.y_trainval)
        X_trainval_bal, y_trainval_bal = Handle_Data.SMOT(r)

        # Standarize data
        scaler = StandardScaler().fit(X_trainval_bal)
        X_trainval_bal_transformed = scaler.transform(X_trainval_bal)
        X_test_transformed = scaler.transform(self.X_test)

        #Train Model
        n_estimators=[100,200]
        max_features=['auto','sqrt','log2']
        min_samples_leaf=[1,25,50]
        best_e_num=0
        best_f_num=''
        best_s_num=0
        best_score=0
        Eva=Evaluation.Evaluation()
        for i in n_estimators:
            for j in max_features:
                for k in min_samples_leaf:
                    RFClassifier=RandomForestClassifier(n_estimators=i,max_features=j,min_samples_leaf=k)
                    scorer = make_scorer(Eva.gini_score, needs_proba=True)
                    score = cross_val_score(RFClassifier, X_trainval_bal_transformed, y_trainval_bal, cv=5,
                                            scoring=scorer)
                    print(score)
                    score = score.mean()
                    print("When parameter n_estimators=",i,", parameter max_features=",j,", parameter min_samples_leaf=",k,":\nMean score is", score)
                    if score >= best_score:
                        best_score = score
                        best_e_num=i
                        best_f_num=j
                        best_s_num=k

        BestRFClassifier = RandomForestClassifier(n_estimators=best_e_num, max_features=best_f_num,
                                                  min_samples_leaf=best_s_num).fit(X_trainval_bal_transformed,
                                                                                   y_trainval_bal)

        #Get the best model
        # BestRFClassifier=RandomForestClassifier(n_estimators=200,max_features='sqrt',min_samples_leaf=1).fit(X_trainval_bal_transformed,y_trainval_bal)
        predict_proba=BestRFClassifier.predict_proba(X_test_transformed)[:,1]
        self.gini=Eva.gini_score(self.y_test,predict_proba)

        return BestRFClassifier,self.gini

    def LDA(self):
        # load data
        df = pd.read_csv('data//train.csv')
        Train_data_transformed = df
        Y = Train_data_transformed["target"]
        X = Train_data_transformed.drop(['target'], axis=1)
        X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, random_state=0)
        # X_train, X_valid, Y_train, Y_valid = train_test_split(X_trainval, Y_trainval, random_state=0)

        # Standarize data
        scaler = StandardScaler().fit(X_trainval)
        X_trainval_transformed = scaler.transform(X_trainval)
        X_test_transformed = scaler.transform(X_test)

        # train LDA model
        Eva = Evaluation.Evaluation()
        best_score = 0
        giniscore = 0
        kfolds = 5
        for C in [10, 20, 30, 40, 50]:
            Data_pca = PCA(n_components=C).fit(X_trainval_transformed)
            X_train_pca = Data_pca.transform(X_trainval_transformed)
            X_test_pca = Data_pca.transform(X_test_transformed)
            lda_model = LinearDiscriminantAnalysis().fit(X_train_pca, Y_trainval)
            prob = lda_model.predict_proba(X_test_pca)[:, 1]
            giniscore = Eva.gini_score(Y_test, prob)
            print("When n_components=",C,":\nMean score is", giniscore)
            if giniscore > best_score:
                best_score = giniscore
                best_parameter = C

        #Get the best model using best parameter we chosen
        # Selected_PCA_model = PCA(n_components=50).fit(X_trainval_transformed)
        Selected_PCA_model = PCA(n_components=best_parameter).fit(X_trainval_transformed)
        X_train_pca_best = Selected_PCA_model.transform(X_trainval_transformed)
        X_test_pca_best = Selected_PCA_model.transform(X_test_transformed)

        LDA_model = LinearDiscriminantAnalysis().fit(X_train_pca_best, Y_trainval)
        self.gini=Eva.gini_score(Y_test, LDA_model.predict_proba(X_test_pca_best)[:, 1])
        return LDA_model,self.gini

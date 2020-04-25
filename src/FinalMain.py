import Modeling
import Data_Cleaning


#Get cleaning data
Clean_Data=Data_Cleaning.DataCleaning()
Clean_Data.Data_Cleaning()
data=Clean_Data.oridata

#Modeling
Model=Modeling.Modeling(data)

## Logistic Regrssion
BestLRModel,gini_coefficiency=Model.Logistic_Regression()
print("The best Logistic Regression Model we train is:",BestLRModel)
print("The Logistic Regression Model has gini-coefficiency:",gini_coefficiency)
print("---------------------------------------------------")

##Random Forest Classifier
BestRFModel,gini_coefficiency1=Model.RandomForest()
print("The best Random forest Model we train is:",BestRFModel)
print("The Random Forest Classifier has gini-coefficiency:",gini_coefficiency1)
print("---------------------------------------------------")

##Neural Network
BestMLPModel,gini_coefficiency2=Model.Neural_Network()
print("The best Neural Network Model we train is:",BestMLPModel)
print("The Neural Network Model has gini-coefficiency:",gini_coefficiency2)
print("---------------------------------------------------")
#
# ##LDA
# BestLDAModel,gini_coefficiency3=Model.LDA()
# print("The best LDA model we train is:",BestLDAModel)
# print("The LDA Model has gini-coefficiency:",gini_coefficiency3)
# print("---------------------------------------------------")






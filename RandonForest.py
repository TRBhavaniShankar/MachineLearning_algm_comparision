import numpy as np
import pandas as pd
from scipy import stats

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

trainingDataSet = pd.read_csv("A3_training_dataset.tsv",delimiter="\t", header=None)
testDataSet = pd.read_csv("A3_test_dataset.tsv",delimiter="\t", header=None)

classLabel = trainingDataSet.iloc[:,-1]
trainingCorrData = trainingDataSet.iloc[:,:-1]

correlation, pValue = stats.spearmanr(trainingCorrData)
columns = np.full((correlation.shape[0],), True, dtype=bool)
for i in range(correlation.shape[0]):
    for j in range(i + 1, correlation.shape[0]):
        if correlation[i, j] > 0:  # Features below this threshold value are eliminated
            if columns[j]:
                columns[j] = False

columns_Selected = trainingCorrData.columns[columns]
print("Columns selected:",len(columns_Selected))
trainingData = pd.DataFrame(trainingCorrData[columns_Selected])
trainingData.insert(loc=len(columns_Selected), column="class",value=classLabel)
testData = pd.DataFrame(testDataSet[columns_Selected])

trainingData = trainingData.astype(float).values.tolist()
trainingData = pd.DataFrame(trainingData)
X = trainingData.iloc[:,:-1]
Y = trainingData.iloc[:,-1]

model = RandomForestClassifier()
model.fit(X,Y)


param_grid = {
    'n_estimators': [900, 1800, 2100],
    'min_samples_split': [1.0, 10, 20],
    'min_samples_leaf' : [0.5, 5, 10],
    'max_depth' : [10,12,15]
}
cv = StratifiedKFold(5,shuffle=True)


grid1 = GridSearchCV(model,param_grid=param_grid,cv=cv, scoring='precision')

grid1.fit(X,Y)
print("Tuning parameters for precision")
print("The parameters combination that would give best accuracy is : ")
print(grid1.best_params_)
print("Grid scores:")

meanPrecScore = grid1.cv_results_['mean_test_score']
precRank = grid1.cv_results_['rank_test_score']

std_Prec = grid1.cv_results_['std_test_score']

DeviationPrecision = np.take(std_Prec, np.argmin(precRank))

print("Standard Deviation of precision: ", DeviationPrecision)

print('-----------------------------------------------------------------------------')

grid2 = GridSearchCV(model,param_grid=param_grid,cv=cv, scoring='recall')

grid2.fit(X,Y)
print("Tuning parameters for recall")
print("The parameters combination that would give best accuracy is : ")
print(grid2.best_params_)
print("Grid scores:")


meanRecScore = grid2.cv_results_['mean_test_score']
recRank = grid2.cv_results_['rank_test_score']

std_rec = grid2.cv_results_['std_test_score']

DeviationRecall = np.take(std_rec, np.argmin(recRank))

print("Standard Deviation of Recall: ", DeviationRecall)
# print("std Recall: ", std_rec)


plt.fill_between(meanRecScore, meanPrecScore, step="pre")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# modelChosen = KNeighborsClassifier(n_neighbors=8, metric="minkowski", weights="uniform")
# modelChosen.fit(X,Y)
#
# prob = modelChosen.predict_proba(testData)

# probAvg = np.average(prob)
# print("Average prediction Probability: ", probAvg)
# print("Writing probabilities to file")
# file = open("RandomForestProb.txt","w+")
# probClass1 = prob[:,-1]
# for i in probClass1:
#     file.write(str(i))
#     file.write("\n")
# file.close()
# #print(prob.iloc[:,-1])

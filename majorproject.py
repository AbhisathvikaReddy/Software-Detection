import pandas as pd
import numpy as np
data = pd.read_csv('projectdataset.csv')
data

import matplotlib.pyplot as plt
data = pd.read_csv('projectdataset.csv')

label = data.groupby('defects').size()
label.plot(kind="bar")
plt.xlabel("Software Defects 0 (Defects),' 1 (Non-Defects)")
plt.ylabel("Count")
plt.title("Software Defects Graph")
unique, count = np.unique(data['defects'], return_counts=True)
print("Number of Non-Defects : "+str(count[0]))
print("Number of Defects : "+str(count[1]))
plt.show()
plt.savefig("dataset_before_smote.jpeg")

from imblearn.over_sampling import SMOTE
import pandas as pd
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
df_resampled.columns = data.columns
data=df_resampled
original=data
data

label = data.groupby('defects').size()
label.plot(kind="bar")
plt.xlabel("Software Defects 0 (Defects),' 1 (Non-Defects)")
plt.ylabel("Count")
plt.title("Software Defects Graph")
unique, count = np.unique(data['defects'], return_counts=True)
print("Number of Non-Defects : "+str(count[0]))
print("Number of Defects : "+str(count[1]))
plt.show()
plt.savefig("dataset_after_smote.png")

print("After oversampling:\n", df_resampled.iloc[:,-1].value_counts())

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
skb = SelectKBest(chi2, k=15)
X_chi2 = skb.fit_transform(X, y)
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SFS(knn,
          k_features=12,
          forward=False,
          floating=False,
          verbose=2,
          scoring='accuracy',
          cv=5)
sbs.fit(X_chi2, y)
X_sbs = X.iloc[:, list(sbs.k_feature_idx_)]

print("SBS selected features: ", X_sbs.columns)
print("Number of features beforefeature selection:",original.shape[1]-1)
selected_feature_names = X_sbs.columns
X_selected_df = pd.DataFrame(X_sbs, columns=selected_feature_names)
print("Number of features after feature selection:",X_selected_df.shape[1])

data = X_selected_df
data
data = pd.concat([pd.DataFrame(data), pd.DataFrame(y)], axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data[['loc','v(g)','ev(g)','n','v','d','i','e','b','t','locomment','loblank']],data['defects'],test_size= 0.2)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

precision = []
recall = []
fscore = []
accuracy = []
def calculateMetrics(algorithm, predict, testY):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    print()
    print(algorithm+' Accuracy  : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' FSCORE      : '+str(f))
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    labels = ['Non-Defects', 'Defects']
    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize =(5, 5))
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

rf=RandomForestClassifier()
rf.fit(x_train,y_train)
rf_prob=rf.predict(x_test)
calculateMetrics("Random Forest",rf_prob,y_test)
rfac_mse = mean_squared_error(y_test, rf_prob.astype(int))
print("Random forest MSE:", rfac_mse)

lr = LogisticRegression(random_state=42)
lr.fit(x_train, y_train)
lr_prob = lr.predict(x_test)
calculateMetrics("Logistic Regression", lr_prob, y_test)
log_mse = mean_squared_error(y_test, lr_prob.astype(int))
print("logistic regression MSE:", log_mse)

from sklearn.metrics import confusion_matrix

X_combined = np.column_stack((rf_prob, lr_prob))
reg = LinearRegression()
reg.fit(X_combined, y_test)
y_pred = reg.predict(X_combined)
from sklearn.metrics import accuracy_score, mean_squared_error
ensemble_acc = accuracy_score(y_test, y_pred.round())
ensemble_mse = mean_squared_error(y_test, y_pred)
print("Ensemble Accuracy:", ensemble_acc)
print("Ensemble MSE:", ensemble_mse)

# Step 1: Calculate the confusion matrix
y_pred_binary = (y_pred >= 0.5).astype(int)
ensemble_cm = confusion_matrix(y_test, y_pred_binary)

# Step 2: Calculate precision, recall, and F-measure
tp = ensemble_cm[1][1]
fp = ensemble_cm[0][1]
fn = ensemble_cm[1][0]

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f_measure = 2 * (precision * recall) / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("F-measure:", f_measure)

labels = ['Non-Defects', 'Defects']
conf_matrix = ensemble_cm
plt.figure(figsize =(5, 5))
ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
ax.set_ylim([0,len(labels)])
plt.title("Ensemble model Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
print("Ensemble Accuracy:", ensemble_acc)
print("Ensemble MSE:", ensemble_mse)
print("Precision:", precision)
print("Recall:", recall)
print("F-measure:", f_measure)
plt.show()

import pickle
with open('model1.pkl','wb') as file:
  pickle.dump(rf,file)
with open('model2.pkl','wb') as file:
  pickle.dump(lr,file)
with open('model3.pkl','wb') as file:
  pickle.dump(reg,file)
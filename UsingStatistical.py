import csv
import pandas as  pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import naive_bayes

rows1=[]

with open('features.txt', 'r') as csvfile:
                csvreader = csv.reader(csvfile,delimiter=' ')
            #fields = csvreader.next()
                for row in csvreader:
                    if('mean' in row[1]):
                        rows1.append(int(row[0])-1)
                    elif('std' in row[1]):
                        rows1.append(int(row[0])-1)
                        
                    #rows1.append(row[:-1])
                    #outp.append(row[-1])
            #X=np.array(rows1, dtype=np.float32)
            #y=np.array(outp, dtype=np.float32)

print(len(rows1))

def process_data(df):
    data=[]
    for i in range(df.shape[0]):
        t=df.iloc[i]
        t=t[0].split()
        t=[ float(j) for j in t]
        data.append(t)  
    data=np.array(data)
    return data

dataframe1=pd.read_csv("X_train.txt")
xtrain=process_data(dataframe1)
datafr=pd.read_csv("y_train.txt")
ytrain=datafr.to_numpy()
datafram3=pd.read_csv("X_test.txt")
xtest=process_data(datafram3)
ytest=pd.read_csv("y_test.txt")
ytest=ytest.to_numpy()

#print(len(xtrain))
#print(len(xtrain[59]))

X_tra=[]
X_test=[]
#Y_train=[]
#Y_test=[]

for i in range(0,len(xtrain)):
    modrow=[]
    for j in range(0,len(rows1)):
        modrow.append(float(xtrain[i][rows1[j]]))
    X_tra.append(modrow)
    
for i in range(0,len(xtest)):
    modrow=[]
    for j in range(0,len(rows1)):
        modrow.append(float(xtest[i][rows1[j]]))
    X_test.append(modrow)

    


#clf=svm.SVC(kernel='rbf',C=0.1,gamma=0.5)
#clf=LogisticRegression()
#clf = RandomForestClassifier()
clf=KNeighborsClassifier()
#clf =tree.DecisionTreeClassifier()
#clf=naive_bayes.GaussianNB()
clf.fit(X_tra,ytrain)
ypred=clf.predict(X_test)
print("Naive bayes",accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest, ypred))
#print(len(rows1))
#print(rows1)


def process_data(df):
    data=[]
    for i in range(df.shape[0]):
        t=df.iloc[i]
        t=t[0].split()
        t=[ float(j) for j in t]
        data.append(t)  
    data=np.array(data)
    return data
import pandas as  pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.manifold  import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
dataframe1=pd.read_csv("X_train.txt")
xtrain=process_data(dataframe1)
datafr=pd.read_csv("y_train.txt")
ytrain=datafr.to_numpy()
datafram3=pd.read_csv("X_test.txt")
xtest=process_data(datafram3)
ytest=pd.read_csv("y_test.txt")
ytest=ytest.to_numpy()

##
##
##principaldf=pd.DataFrame(data=X, columns = ['component1','component2'])
##targetdf=pd.DataFrame(data=ytest,columns=['target'])
##df4=pd.concat([principaldf,targetdf],axis=1)
##ax=sns.scatterplot(x='component1',y='component2',data=df4,hue='target',palette="ch:r=-.5,l=.75").plot()
###legend_labels, _=ax.get_legend_handles_labels()
##plt.legend(['walking','walking upstairs','walking_downstairs','sitting','standing','laying'],bbox_to_anchor=(1,0.5),title='type')
##print(ax)
##plt.show()


#KernelPCA
transform = KernelPCA(n_components=50,kernel='poly')
#transform = PCA(100)

Xtrain=transform.fit_transform(xtrain)
Xtest=transform.fit_transform(xtest)
#clf =svm.SVC(kernel='rbf',gamma='scale')
#clf = GaussianNB()
#clf=LogisticRegression()
#clf = RandomForestClassifier()
clf=KNeighborsClassifier()
####ypredic = clf.predict(Xtest)
#clf =tree.DecisionTreeClassifier()
#clf=clf.fit(Xtrain,ytrain)
##tree.plot_tree(clf)
clf.fit(Xtrain,ytrain)

ypred=clf.predict(Xtest)
print("normal svm",accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))







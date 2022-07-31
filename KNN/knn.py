#K Nearest Neibors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

#------read the data:
df = pd.read_csv('teleCust1000t.csv')
# print(df.head().to_string())

#------Let’s see how many of each class is in our data set:
# print(df['custcat'].value_counts())

#-----show histogram:
# df.hist(column='income', bins=50)

#-----print the columns of df:
# print (df.columns)

#----- تبدیل دیتا به آرایه برای تحلیل:
X = df[['region', 'tenure','age', 'marital', 'address', 'income',
        'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)

y = df['custcat'].values

#------Normalize Data------:
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

#------Train Test Split------:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)

#------K Nearest Neighbor (KNN)------:
from sklearn.neighbors import KNeighborsClassifier 
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

#------predicting------:
yhat = neigh.predict(X_test)
print(yhat[0:5])

#------Accuracy evaluation------:
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

#------What about other K?
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

# print (mean_acc)

#------Plot the model accuracy for a different number of neighbors.
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.savefig("figure.png")
plt.show()


# print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 






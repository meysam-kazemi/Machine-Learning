#house price
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("D:/0.py/machine learning/project/house price/housePrice.csv")
print(df.head().to_string())


# df.info()
df.dropna(inplace=True)

df.Address.value_counts()
address = np.unique(np.asarray(df.Address))

import sklearn.preprocessing as prep

label_i=prep.LabelEncoder()
label_i.fit(address)
df.Address=label_i.transform(df.Address)

"""data cleaning"""

df.dropna(inplace=True)
Area_filter=df['Area'].str.isnumeric()
df=df[Area_filter]
df['Area']=df['Area'].astype(int)

df.corr()

"""convert the data to array"""
x=df[["Area" , "Room"]].values
y=df[["Price(USD)"]].values




df.Room.hist()
plt.scatter(x[:,0],y,c='red')

"""train and test"""
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(x,y,test_size=0.2)

from sklearn import linear_model 
regr=linear_model.LinearRegression()
regr.fit(Xtrain,Ytrain)

from sklearn.metrics import r2_score
predict=regr.predict(Xtest)

r2=r2_score(Ytest,predict)

print(r2)


"""my house price"""
# myX=[[70,3]]
# z=regr.predict(myX)
# print(z)




class regression():
    def __init__(self,df_=0,X=0,Y=0):
        try:
            self.X=np.asanyarray(df_[["Area" , "Room"]])
            self.Y=np.asanyarray(df_[["Price(USD)"]])
            print("-DataFrame input-")
        except:
            self.X = X
            self.Y = Y
            print("-numpy input-")
        self.address = address
        
        self.regr=linear_model.LinearRegression()
        self.regr.fit(self.X,self.Y)
        self.regr.coef_
        
        print("-"*5,"model fitted","-"*5)
        
    def mymodel(self,address,**kwards):
        self.address = address

        try:
            self.theta0 = dic[str(self.address)][0]
            self.theta1 = dic[str(self.address)][1]
            self.theta2 = dic[str(self.address)][2]
        except:
            self.theta0 = dic["under_20"][0]
            self.theta1 = dic["under_20"][1]
            self.theta2 = dic["under_20"][2]
            
    def predict(self,Xtest):
        return self.theta0*Xtest[:,0] + self.theta1*Xtest[:,1]+self.theta2
    def r2_score(self,Xtest,Ytest):
        return r2_score(Ytest,self.predict)

    def thetas(self):
        theta0 = self.regr.coef_[0][0]
        theta1 = self.regr.coef_[0][1]
        theta2 = self.regr.intercept_[0]
        return [theta0,theta1,theta2]
     
    def Address_label(self,Address,Label_Encoder=label_i):
        test = np.array([Address])
        return label_i.transform(test)
    
        
address = np.unique(np.asarray(df.Address))

dic = {}
for a in address:
    df_copy = df[df.Address==a]
    if df_copy.Area.count()>20:
        rg = regression(df_copy)
        dic[f"{a}"]=rg.thetas()
        
under_20 = df[df.Address<=20]
rg = regression(under_20)
dic["under_20"]=rg.thetas()


x=df[["Area" , "Room","Address"]].values
y=df[["Price(USD)"]].values

"""train and test"""
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(x,y,test_size=0.2)

model = regression(X=Xtrain,Y=Ytrain)
model.mymodel(Xtest[:,2])
pred = model.predict(Xtest[:,0:2])

from sklearn.metrics import mean_squared_error
mean_squared_error(pred,Ytest)


r2_score(Ytest,pred)




import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

train_name="D:/0Course/AN6001 AI and Big Data in Business/Group Project/report/normalized data without employ_source.csv"

data = pd.read_csv(train_name,encoding='utf8',engine='python')
data = data.dropna()

# In[6]:
# In[12]:


data_train,data_test = train_test_split(data,test_size=0.3,random_state=4)


# In[13]:


Xtrain = data_train.drop(columns=['Attrition'])
Ytrain = data_train.loc[:, ["Attrition"]]
Xtest = data_test.drop(columns=['Attrition'])
Ytest = data_test.loc[:, ["Attrition"]]



#Neural Network Model:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model=Sequential()
model.add(Dense(128, input_dim=len(Xtrain.columns), activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(64, input_dim=len(Xtrain.columns), activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(32, input_dim=len(Xtrain.columns), activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(16, input_dim=len(Xtrain.columns), activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(8, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(4, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'Adamax', metrics = ['accuracy'])
model.fit(Xtrain, Ytrain, batch_size = 10, epochs=100, verbose=1)

#Evaluate the NNET model
print(model.summary())
print(model.evaluate(Xtrain, Ytrain))
print(model.evaluate(Xtest, Ytest))

#Calculate the accuracy for train data
pred=model.predict(Xtrain)
pred=np.where(pred>0.5,1,0)
cm=confusion_matrix(Ytrain, pred)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
print(f'The accuracy for train data in NNET is {round(accuracy*100,2)}%')

#Calculate the accuracy for test data
pred=model.predict(Xtest)
pred=np.where(pred>0.5,1,0)
cm=confusion_matrix(Ytest, pred)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
print(f'The accuracy for test data in NNET is {round(accuracy*100,2)}%') 


#XG Boost
from sklearn.ensemble import GradientBoostingClassifier


# In[26]:


model = GradientBoostingClassifier(max_depth=8)
model.fit(Xtrain, Ytrain)
pred = model.predict(Xtrain)

#Calculate the accuracy for train data
pred = model.predict(Xtrain)
cm = confusion_matrix(Ytrain, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(f'The accuracy for train data in XG Boost is {round(accuracy*100,2)}%')

#Calculate the accuracy for test data
pred = model.predict(Xtest)
cm = confusion_matrix(Ytest, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(f'The accuracy for test data in XG Boost is {round(accuracy*100,2)}%')

#check XGBoost feature importance
def feature_imp(df, model):
    fi = pd.DataFrame()
    fi["feature"] = df.columns
    fi["importance"] = model.feature_importances_
    return fi.sort_values(by="importance", ascending=True)

df = feature_imp(Xtrain, model)
df.set_index('feature', inplace=True)
df.plot(kind='barh', figsize=(10, 20))
plt.title('Feature Importance according to XGBoost')
plt.show()


'''
#LSTM
from tensorflow.keras.layers import LSTM

data_size = len(data)
print(data_size)

timesteps = 3 84 50
df_resize = int(data_size//timesteps)
print(df_resize)

df_trunc = df_resize * timesteps
print(df_trunc)
type(df_trunc)

X = data.drop(columns=['Attrition'])
Y = data.loc[:, ["Attrition"]]

df_dim = len(X.columns)
print(df_dim)

X = X.values
Y= Y.values

print(type(X))

X.shape

X = X[:df_trunc,:]
Y = Y[:df_trunc]
X.shape
print(Y)
Y.shape

X.shape

X_shaped = X.reshape(df_resize,timesteps,df_dim)
Y_shaped = Y.reshape(df_resize, timesteps)

print(X_shaped.shape)
print(Y_shaped.shape)

Y_shaped = Y_shaped[:,-1]

print(Y_shaped)

X_train, X_test, Y_train, Y_test = train_test_split(X_shaped, Y_shaped, test_size=0.3, random_state=1)
model = Sequential()
model.add(LSTM(34,return_sequences = True, input_shape=(timesteps, df_dim)))
model.add(Dropout(0.2))
model.add(LSTM(30,return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(20,return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(10))
model.add(Dropout(0.2))
model.add(Dense(1, activation = "sigmoid"))
model.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])
model.fit(X_train, Y_train, batch_size=10, epochs=100)

model.evaluate(X_test, Y_test)

pred=model.predict(X_test)
pred=np.where(pred>0.5,1,0)
cm=confusion_matrix(Y_test, pred)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
print(f'The accuracy for test data in LSTM is {round(accuracy*100,2)}%')


# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# optimisation for decision tree
# max_depth = 3 gives the best accuracy
for i in range(20):
    print(i)
    model = DecisionTreeClassifier(max_depth=i+1)
    model.fit(Xtrain, Ytrain)
    pred = model.predict(Xtest)
    cm = confusion_matrix(Ytest, pred)
    accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
    print(accuracy)


# Decision Tree Model
model = DecisionTreeClassifier(max_depth=3)
model.fit(Xtrain, Ytrain)
pred = model.predict(Xtest)

# Confusion Matrix on testset - DT
cm = confusion_matrix(Ytest,pred)
print(cm)
# [[357  12]
# [ 51  21]]

accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(f'The accuracy for test data in Decision Tree is {round(accuracy*100,2)}%')
# 85.71%

# Feature importance - DT
def feature_imp(df, model):
    fi = pd.DataFrame()
    fi["feature"] = df.columns
    fi["importance"] = model.feature_importances_
    return fi.sort_values(by="importance", ascending=False)

df = feature_imp(Xtrain, model)
df.set_index('feature', inplace=True)
df.plot(kind='barh', figsize=(18, 14))
plt.title('Feature Importance according to Decision Tree')
plt.show()
# CHECK: why only top few variables plotted?
# Total Working Years, Monthly Income, Overtime, Hourly Rate, Age


#%%
# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=3)
model.fit(Xtrain, Ytrain)
pred = model.predict(Xtest)

# Confusion Matrix on testset - RF
cm = confusion_matrix(Ytest, pred)
print(cm)
# [[369   0]
# [ 69   3]]

accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(f'The accuracy for test data in Random Forest is {round(accuracy*100,2)}%')
# 84.35%

# Feature importance - RF
def feature_imp(df, model):
    fi = pd.DataFrame()
    fi["feature"] = df.columns
    fi["importance"] = model.feature_importances_
    return fi.sort_values(by="importance", ascending=False)

df = feature_imp(Xtrain, model)
df.set_index('feature', inplace=True)
df.plot(kind='barh', figsize=(18, 14))
plt.title('Feature Importance according to Random Forest')
plt.show()'''





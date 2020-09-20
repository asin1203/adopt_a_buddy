import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle

train = pd.read_csv('D:\\pypy\\train.csv')
train_data=train.drop("pet_id",axis=1)
train_data["condition"]=train_data["condition"].fillna(-1)
le = preprocessing.LabelEncoder()
train_data['ctype']=le.fit_transform(train_data['color_type'])
train_data['issue_date']=pd.to_datetime(train_data['issue_date'])
train_data['listing_date']=pd.to_datetime(train_data['listing_date'])
train_data['duration']=abs(train_data['listing_date']-train_data['issue_date']).dt.days

x=train_data[['condition','ctype','length(m)','height(cm)','X1','X2','breed_category','duration']]
x=x.to_numpy()
y=train_data['pet_category']

x_train, x_test, y_train, y_test =  train_test_split(x,y,test_size=0.1, random_state=1)

my_model = xgb.XGBClassifier()

parameters={
    "eta" : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth" : [3,4,5,6],
    "min_child_weight" : [1,3,5],
    "gamma" : [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree" : [0.5, 0.7, 1]
}
randCV = RandomizedSearchCV(my_model, parameters, n_iter=10, refit= True, cv=3, n_jobs=2)
randCV.fit(x_train, y_train)
y_pred= randCV.predict(x_test)

ac2 =  accuracy_score(y_test, y_pred)
f12=f1_score(y_test, y_pred, average="weighted")

with open("model2.pkl","wb") as f:
 pickle.dump(randCV,f)
print("Accuracy score after tuning {}".format(ac2))
print("F1 score after tuning {}".format(f12))



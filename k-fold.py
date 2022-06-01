import pandas as pd
data = pd.read_csv('diabetics.csv')

x = data.drop('outcome', axis=1)
y = data['outcome']


from sklearn.model_selection import KFold  # library kfold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


                            
#    هيقسم الداتا الي 5 اجزاء و هيمشي علي الداتا 5 مرات
k = 5
kfold = KFold(n_splits=k, random_state=None)

acc_list = []

# train_index , test_index  ==> data في ال row هو رقم ال 
# kf.split(X) ==> train_index , test_index الي  data هتقسم ال
# k=5 ديه هتتنفذ 5 مرات علشان ال  loop ال 
for train_index , test_index in kfold.split(x):
    x_train , x_test = x.iloc[train_index , :] , x.iloc[test_index , :]   # روح للصف الي انا محدده وهاتلي كل الي فيه
    y_train , y_test = y[train_index] , y[test_index]
    
    rf = RandomForestClassifier(n_estimators = 10)  #number tree it will build to return data
    
    rf.fit(x_train , y_train)  # here learn an algorith
    
    predictions = rf.predict(x_test)  # this here exam an algorithm
    
    acc = accuracy_score(predictions , y_test)
    
    acc_list.append(acc)  # add accuracy in list


avg_acc_list = sum(acc_list)/k     
    
print('accuracy of each fold ' , acc_list)
print('Avg accuracy : ' , avg_acc_list)


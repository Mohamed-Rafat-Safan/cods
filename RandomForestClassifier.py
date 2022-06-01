import pandas as pd   # use to read dataset
data = pd.read_csv('diabetics.csv')


data.shape  # to print number of raw and column
# data.head()


x = data.drop('outcome', axis=1)  # axis=1 ==> كعمود لو مسحتها هيروح يدور في الصفوف outcome ديه معناها انك بتقوله روح ابحث عن ال 
y = data['outcome']


# hold out method  ==>   dataset طريقة لتقسيم ال 
from sklearn.model_selection import train_test_split #  library    train and test هتقسم الداتا ال 

# this divid the data to training and  testing  (x , y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # train = 0.7

# print(data.head())


from sklearn.ensemble import RandomForestClassifier  

rf = RandomForestClassifier(n_estimators = 10)  #number tree it will build to return data
rf.fit(x_train , y_train)  # here learn an algorith ===> (model)


predictions = rf.predict(x_test)  # this here exam an algorithm
# y_test الي هو الامتحان بي  prediction بعد كده هقارن 

from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test , predictions)

print(matrix)

# tn  fp
# fn  tp


from sklearn.metrics  import accuracy_score

acc = accuracy_score(y_test,predictions)  # هنا بحسب الدقة

print("accuracy " , acc)

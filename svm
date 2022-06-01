import pandas as pd
from sklearn import svm
data = pd.read_csv('test.csv')

x = data.drop('outcome', axis=1)
y = data['outcome']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

classifiere = svm.SVC(kernel="rbf")

classifiere.fit(x_train , y_train)


predictions = classifiere.predict(x_test)  # this here exam an algorithm
# y_test الي هو الامتحان بي  prediction بعد كده هقارن 

from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test , predictions)

print(matrix)


from sklearn.metrics  import accuracy_score
acc = accuracy_score(y_test,predictions)
print("accuracy " , acc)


#precision
from sklearn.metrics import precision_score
pre=precision_score(y_test,predictions)
print("precision ", pre)

#recall 
from sklearn.metrics import recall_score
rec=recall_score(y_test,predictions)
print("recall  ",rec )

#f1-measure
# recall وال precision العلاقه بين ال 
from sklearn.metrics import f1_score
f1=f1_score(y_test,predictions)
print("f1-measure  ",f1 )

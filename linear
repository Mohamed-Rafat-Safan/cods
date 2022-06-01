import pandas as pd
import numpy as np   # error لكي اجيب الجزر بتاع 
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')
dataset.shape
dataset.head()


x = dataset.iloc[ : , : -1].values
y = dataset.iloc[ : , 1].values 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
print(x_test)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train , y_train) # learn algorithm

# y = mx + a  where m==>slop , a==>interception with y

print("Intercept ",regressor.intercept_)
print("Coef ",regressor.coef_)

y_pred = regressor.predict(x_test)

df = pd.DataFrame({'Actual':y_test , 'Predicted':y_pred})
print(df)


from sklearn import metrics

print('Mean Absolute Error: ' , metrics.mean_absolute_error(y_test , y_pred))  # هنا هيطبع نسبة الخطأ بين القيمة الحقيقية والقيمة المتوقعة
print('Mean Squared Error: ' , metrics.mean_squared_error(y_test , y_pred))
print('Root Mean Absolute Error: ' , np.sqrt(metrics.mean_squared_error(y_test , y_pred)))

dataset.plot(x='Hours' , y='Scores' , style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.plot(x_test , y_pred , color="blue", linewidth=3)
plt.show()

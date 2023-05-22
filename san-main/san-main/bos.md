# Importing Necesarry Packages
import pandas as pd
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore") 

df = pd.read_csv("HousingData.csv")
df.head()

df.shape

df.dtypes

df.nunique()

df.isnull().sum()

df[df.isnull().any(axis=1)]

corr=df.corr()
corr.shape


plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15},cmap='Greens')

updated_df=df.dropna(axis=1)
updated_df.info()
x=updated_df.drop(['MEDV'],axis=1)
y=updated_df['MEDV']


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3, random_state=4)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train, y_train)

lm.intercept_

coefficients=pd.DataFrame([x_train.columns, lm.coef_]).T
coefficients=coefficients.rename(columns={0:'Attribute',1:'coefficients'})
coefficients

#Model Evaluation
y_pred=lm.predict(x_train)

#model Evaluation
print('R^2:', metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:', 1-(1-metrics.r2_score(y_train, y_pred)*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1)))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

plt.scatter(y_train, y_pred)
plt.xlabel("prices")
plt.ylabel("predicted Prices")
plt.title("prices vs predicted prices")
plt.show()

plt.scatter(y_pred, y_train-y_pred)
plt.title("predicted vs Residual")
plt.xlabel("predicted")
plt.ylabel("Residuals")
plt.show()

sns.distplot(y_train-y_pred)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

#for test Data
y_test_pred=lm.predict(x_test)

#model Evaluation
acc_linreg=metrics.r2_score(y_test, y_test_pred)
print('R^2:', acc_linreg)
print('Adjusted R^2:', 1-(1-metrics.r2_score(y_test, y_test_pred)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))





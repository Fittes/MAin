import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df.boxplot("word_count")
--------
ax = sns.boxplot(x=result["word_count"])
--------------
quant = result["word_count"].quantile(0.6)# считаем квантиль 90 %
quant_low = result["word_count"].quantile(0.01)# считаем квантиль 10 %
df2 = result[result["word_count"] < quant] # убираем выбросы по квантилю
df2 = df2[df2["word_count"] > quant_low] # убираем выбросы по квантилю
-------------------
ax1 = sns.boxplot(x=df2['word_count'])
------------------------
X = df2.drop("word_count", axis = 1) #Удаляем столбец с выборкой ответо из основного df
y = df2["word_count"] #Создаём выборку ответов
--------------------
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
----------------
train_test_split
-------------
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, learning_curve
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Lasso, Ridge
--------------------
X_df = result.drop('word_count', axis=1)
------------------
y = result['word_count']
-------------------
X_train, X_valid, y_train, y_valid = train_test_split(X_df, y, test_size=0.2, random_state=42)
print(X_train.shape, X_valid.shape)
--------------------
pd.Series(y_train).hist()
pd.Series(y_valid).hist()
---------------------
from sklearn import linear_model
reg = linear_model.LassoLars(alpha=.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
LassoLars(alpha=0.1)
reg.coef_
#пошёл бред
y = result['word_count']

X_df = result.drop('word_count', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_df, y, test_size = 0.3, random_state=42)
print(X_train.shape, X_valid.shape)



https://scikit-learn.org/stable/modules/linear_model.html
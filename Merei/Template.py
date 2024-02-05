import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("train.csv")
model = LinearRegression()

X = data[data.columns.drop("target")]
y = data["target"]
model.fit(X,y)
# эта модель должна теперь принимать данные которые занинуты через интерфейс
# ниже просто пример как это будет происходить

x = [[41, 1, 1041834, 611490, 1655138, 2629221, 2514074, 3118, 108, 6, 105, 4, 6, 1, 102, 83, 41033, 270685, 2, 4, 2, 54420, 8]]
# x это просто данные про одного человека
print("Вероятность его дефолта:", model.predict(x)[0])
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
import pickle


data = pd.read_csv('Salary_Data.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

# print(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print("Shape of training data: ", X_train.shape)
print("Shape of test data: ", X_test.shape)

# algorithm = LogisticRegression(solver='liblinear', multi_class='auto')
# algorithm = LinearRegression()
algorithm = MLPClassifier(solver='lbfgs', alpha=0.5, hidden_layer_sizes=(5, 2), max_iter=1000, random_state=1)

algorithm.fit(X_train, y_train)

y_predict = algorithm.predict(X_test)
print(y_predict, y_test)

score = algorithm.score(X_test, y_test)
print("Score:", score)

rms = mean_absolute_error(y_test, y_predict)
print("RMS:", rms)

pickle.dump(algorithm, open("model.pkl", "wb"))

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[1.8]]))

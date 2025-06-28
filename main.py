from utils import load_data, train_model, evaluate_model

X_train, X_test, y_train, y_test = load_data("digits.csv")
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)

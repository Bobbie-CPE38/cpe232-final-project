from sklearn.metrics import mean_absolute_error

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return {"mae": mae}
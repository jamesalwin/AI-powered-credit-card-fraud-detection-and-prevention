from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_model(X, y, model_type='logistic'):
    if model_type == 'logistic':
        model = LogisticRegression(solver='liblinear')
    else:
        raise ValueError("Unsupported model_type")

    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    return report, matrix, roc_auc

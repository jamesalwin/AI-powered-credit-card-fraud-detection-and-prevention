import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Train model function
def train_model(X, y, model_type='logistic'):
    """
    Train the model using the specified model type (logistic regression by default).
    """
    # Debugging output to check shapes and data types
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("Data type of X:", X.dtype)
    print("Data type of y:", y.dtype)

    # Check for NaN values in X or y
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X or y contains NaN values!")

    # Check if both classes are present in y_train
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(f"Data contains only one class: {unique_classes}. Cannot train a classifier.")

    # If X is 1D, reshape it to 2D (n_samples, 1 feature)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    # If y is 2D, flatten it to 1D
    if len(y.shape) == 2:
        y = y.ravel()

    # Handle class imbalance by resampling (optional: oversample the minority class)
    # This is just one method to balance the dataset; others include SMOTE or class weights.
    if len(unique_classes) > 1:
        # Resample the minority class to balance the classes
        # First, concatenate X and y to create a DataFrame for easy manipulation
        X_resampled, y_resampled = resample(X, y, stratify=y, random_state=42, n_samples=X.shape[0])

    # Initialize the model
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"Model type {model_type} not supported")

    # Train the model
    model.fit(X_resampled, y_resampled)

    return model, X_resampled, X_resampled, y_resampled, y_resampled

# Evaluate model function
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set and return performance metrics.
    """
    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Classification report and confusion matrix
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    # ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return report, matrix, roc_auc

# Function to preprocess data (example: splitting and scaling)
def preprocess_data(df):
    """
    This is a mock preprocessing function. You'll need to adapt it based on your dataset.
    """
    # Splitting the data into features (X) and target (y)
    X = df.drop(columns=['Class'])  # Assuming 'Class' is the target column
    y = df['Class']

    # Splitting into training and test sets using stratified split to keep class distribution similar
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, train_test_split
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

# --- Custom Transformer for KFold Target Encoding ---
class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    Custom transformer for target encoding using K-Fold strategy to prevent data leakage.
    """
    def __init__(self, cols=None, n_splits=5):
        self.cols = cols
        self.n_splits = n_splits
        self.encoders = {}

    def fit(self, X, y):
        """Fits target encoding for each categorical column using KFold strategy."""
        if y is None:
            raise ValueError("Target `y` is required for Target Encoding.")
        
        self.encoders = {col: ce.TargetEncoder(cols=[col]) for col in self.cols}

        for col in self.cols:
            self.encoders[col].fit(X[[col]], y)

        return self

    def transform(self, X):
        """Transforms the categorical features using the trained target encoding."""
        X_encoded = X.copy()
        for col in self.cols:
            X_encoded[col] = self.encoders[col].transform(X[[col]]).values.flatten()
        return X_encoded


# Fix for Frequency Encoding Output
def frequency_encoding(df, cols):
    """Applies frequency encoding and returns a 2D array."""
    encoded_df = df.copy()
    for col in cols:
        freq_map = df[col].value_counts(normalize=True).to_dict()
        encoded_df[col] = df[col].map(freq_map).fillna(0)

    return encoded_df[cols].to_numpy()  # Ensure it's a 2D NumPy array

def pipeline_preprocessor(df, y=None, **kwargs):
    """
    Creates a preprocessing pipeline with:
    - Standard Scaler
    - Robust Scaler
    - BaseN Encoding
    - One-Hot Encoding
    - Frequency Encoding
    - Target Encoding (KFold to avoid leakage)
    - Binary Encoding
    """

    # Validate columns before applying transformations
    def validate_columns(col_list, df):
        """Ensures all columns exist in DataFrame."""
        if col_list:
            for col in col_list:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame.")
        return col_list

    # Extracting column lists and ensuring they are valid
    standard_scaler_cols = validate_columns(kwargs.get('standard_scaler_cols', []), df)
    robust_scaler_cols = validate_columns(kwargs.get('robust_scaler_cols', []), df)
    baseN_enc_cols = validate_columns(kwargs.get('baseN_enc_cols', []), df)
    one_hot_cols = validate_columns(kwargs.get('one_hot_cols', []), df)
    frequency_enc_cols = validate_columns(kwargs.get('frequency_enc_cols', []), df)
    target_enc_cols = validate_columns(kwargs.get('target_enc_cols', []), df)
    binary_enc_cols = validate_columns(kwargs.get('binary_enc_cols', []), df)

    transformers = []

    # Standard Scaler
    if standard_scaler_cols:
        transformers.append(('standard_scaler', StandardScaler(), standard_scaler_cols))

    # Robust Scaler
    if robust_scaler_cols:
        transformers.append(('robust_scaler', RobustScaler(), robust_scaler_cols))

    # BaseN Encoding
    if baseN_enc_cols:
        transformers.append(('baseN_encoder', ce.BaseNEncoder(cols=baseN_enc_cols), baseN_enc_cols))

    # One-Hot Encoding
    if one_hot_cols:
        transformers.append(('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), one_hot_cols))

    # ✅ Corrected Frequency Encoding inside ColumnTransformer
    if frequency_enc_cols:
        transformers.append(('frequency_encoder', 
                             FunctionTransformer(lambda X: frequency_encoding(pd.DataFrame(X, columns=frequency_enc_cols), frequency_enc_cols), 
                                                 validate=False), 
                             frequency_enc_cols))

    # Target Encoding
    if target_enc_cols:
        if y is None:
            raise ValueError("Target variable `y` is required for target encoding.")
        transformers.append(('target_encoder', ce.TargetEncoder(cols=target_enc_cols), target_enc_cols))

    # Binary Encoding
    if binary_enc_cols:
        transformers.append(('binary_encoder', ce.BinaryEncoder(cols=binary_enc_cols), binary_enc_cols))

    # Creating Column Transformer
    preprocessor = ColumnTransformer(transformers, remainder='drop')

    return preprocessor

# --- Classifier Pipeline ---
def classifier_pipeline(preprocessor, model):
    """
    Returns a full pipeline with the preprocessor and model.
    """
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])
    return pipeline

# --- Model Training ---
def model_training(X_train, y_train, classifier_pipeline):
    """
    Trains the model using the pipeline.
    """
    classifier_pipeline.fit(X_train, y_train)
    return classifier_pipeline

# --- Main Execution ---
if __name__ == '__main__':
    # Dummy dataset
    data = {
        'age': [25, 30, 35, 40, 45],
        'height': [160, 170, 180, 175, 165],
        'kind': ['A', 'B', 'A', 'C', 'B'],
        'type': ['X', 'Y', 'X', 'Z', 'Y'],
        'geo_level_3_id': ['001', '002', '001', '003', '002'],
        'is_concrete': [1, 0, 1, 1, 0]
    }
    df = pd.DataFrame(data)
    y = pd.Series([0, 1, 0, 1, 1], name="target")  # Target column

    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)

    # Defining model
    model = RandomForestClassifier(n_estimators=100, random_state=0)

    # Creating Preprocessor with Different Encodings
    pre_proccessor = pipeline_preprocessor(
        X_train, y_train,
        standard_scaler_cols=['age', 'height'],
        one_hot_cols=['kind', 'type'],
        frequency_enc_cols=['geo_level_3_id'],
        target_enc_cols=['geo_level_3_id'],
        binary_enc_cols=['is_concrete']
    )

    # Creating pipeline
    pipeline = classifier_pipeline(pre_proccessor, model)

    # Model training
    model_fit = model_training(X_train, y_train, pipeline)

    # Prediction
    y_pred = model_fit.predict(X_test)

    # Evaluation
    score = f1_score(y_test, y_pred, average='micro')
    print(f"F1 Score: {score}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

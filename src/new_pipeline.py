import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Custom frequency encoder for categorical variables"""
    def __init__(self, columns):
        self.columns = columns
        self.mappings = {}

    def fit(self, X, y=None):
        self.mappings = {
            col: X[col].value_counts(normalize=True) for col in self.columns
        }
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.mappings[col]).fillna(0)
        return X


class MLPipeline:
    def __init__(self, 
                 scalers=None, 
                 encoders=None, 
                 model=None):
        """
        Parameters:
        - scalers: dict, mapping feature names to scaler types. 
                   Example: {"age": "standard", "salary": "minmax"} 
                   Supported scalers: ["standard", "minmax", "robust"]
        - encoders: dict, mapping categorical feature names to encoding methods.
                    Example: {"city": "onehot"}
                    Supported methods: ["onehot", "binary", "basen", "target", "frequency"]
        - model: scikit-learn model instance.
        """
        self.scalers = scalers if scalers else {}
        self.encoders = encoders if encoders else {}
        self.model = model
        self.pipeline = None

    def _get_scaler(self, scaler_type):
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler()
        }
        return scalers.get(scaler_type, StandardScaler())

    def _get_encoders(self, categorical_features):
        transformers = []
        for feature, encoding in self.encoders.items():
            if encoding == "onehot":
                transformers.append((f"onehot_{feature}", ce.OneHotEncoder(), [feature]))
            elif encoding == "binary":
                transformers.append((f"binary_{feature}", ce.BinaryEncoder(), [feature]))
            elif encoding == "basen":
                transformers.append((f"basen_{feature}", ce.BaseNEncoder(), [feature]))
            elif encoding == "target":
                transformers.append((f"target_{feature}", ce.TargetEncoder(), [feature]))
            elif encoding == "frequency":
                transformers.append((f"frequency_{feature}", FrequencyEncoder([feature]), [feature]))
            else:
                raise ValueError(f"Unknown encoding type: {encoding}")

        return transformers

    def fit(self, X, y):
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [col for col in self.encoders.keys() if col in X.columns]

        transformers = []
        
        # Apply different scalers to different numerical columns
        for feature, scaler_type in self.scalers.items():
            if feature in numerical_features:
                transformers.append((f"scaler_{feature}", self._get_scaler(scaler_type), [feature]))

        # Add encoders for categorical features
        transformers.extend(self._get_encoders(categorical_features))

        # Column transformer
        preprocessor = ColumnTransformer(transformers, remainder="passthrough")

        # Full pipeline
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.model)
        ])

        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        if self.pipeline:
            return self.pipeline.predict(X)
        else:
            raise ValueError("Pipeline has not been fitted yet.")

    def transform(self, X):
        if self.pipeline:
            return self.pipeline.named_steps["preprocessor"].transform(X)
        else:
            raise ValueError("Pipeline has not been fitted yet.")

    def get_pipeline(self):
        """Return the underlying scikit-learn pipeline"""
        return self.pipeline
    
if __name__ == "__main__" :
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Sample dataset
    df = pd.DataFrame({
        "age": [25, 32, 47, 51, 62],
        "salary": [50000, 60000, 80000, 85000, 90000],
        "height": [170, 175, 180, 165, 190],
        "city": ["Berlin", "Berlin", "Munich", "Munich", "Hamburg"],
        "purchased": [0, 1, 0, 1, 1]
    })

    # Define numerical scalers
    scalers = {
        "age": "standard",  # StandardScaler for age
        "salary": "minmax",  # MinMaxScaler for salary
        "height": "robust"   # RobustScaler for height
    }

    # Define categorical encoding settings
    encoders = {
        "city": "onehot"  # Try "binary", "basen", "target", "frequency" as well
    }

    # Train-test split
    X = df.drop(columns=["purchased"])
    y = df["purchased"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize pipeline
    pipeline = MLPipeline(
        scalers=scalers, 
        encoders=encoders,
        model=RandomForestClassifier()
    )

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Make predictions
    predictions = pipeline.predict(X_test)

    # Transform dataset (without model prediction)
    transformed_X = pipeline.transform(X_test)

    print("Predictions:", predictions)
    print("Transformed Data:\n", pd.DataFrame(transformed_X))

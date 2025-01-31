import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler,FunctionTransformer
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

class NamedFunctionTransformer(FunctionTransformer):
    """Custom FunctionTransformer that implements get_feature_names_out"""
    def __init__(self, func, inverse_func=None, feature_names_out=None, **kwargs):
        super().__init__(func=func, inverse_func=inverse_func, **kwargs)
        self.feature_names_out = feature_names_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_out
        return [f"{feature}_log" for feature in input_features]

class MLPipeline:
    def __init__(self, scalers=None, encoders=None, model=None):
        """
        Parameters:
        - scalers: dict, mapping feature names to scaler types.
                   Example: {"age": "standard+minmax"}
                   Supported scalers: ["standard", "minmax", "robust"]
        - encoders: dict, mapping categorical feature names to encoding methods.
                    Example: {"city": "onehot+target"} (supports multiple encodings per feature)
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
            "robust": RobustScaler(),
            "log" : NamedFunctionTransformer(np.log1p)
        }
        return scalers.get(scaler_type, StandardScaler())

    def _get_encoders(self, categorical_features):
        """Handles categorical encodings, supporting multiple encodings per feature."""
        transformers = []

        for feature, encoding in self.encoders.items():
            encoding_types = encoding.split("+")  # Split multiple encodings

            for enc in encoding_types:
                new_feature_name = f"{feature}_{enc}"  # Rename feature
                if enc == "onehot":
                    transformers.append((new_feature_name, ce.OneHotEncoder(), [feature]))
                elif enc == "binary":
                    transformers.append((new_feature_name, ce.BinaryEncoder(), [feature]))
                elif enc == "basen":
                    transformers.append((new_feature_name, ce.BaseNEncoder(), [feature]))
                elif enc == "target":
                    transformers.append((new_feature_name, ce.TargetEncoder(), [feature]))
                elif enc == "frequency":
                    transformers.append((new_feature_name, FrequencyEncoder([feature]), [feature]))
                else:
                    raise ValueError(f"Unknown encoding type: {enc}")

        return transformers

    def _get_scalers(self, numerical_features):
        """Handles numerical scalings, supporting multiple scalings per feature."""
        transformers = []

        for feature, scaler_type in self.scalers.items():
            scaling_types = scaler_type.split("+")  # Split multiple scalers

            for scale in scaling_types:
                new_feature_name = f"{feature}_{scale}"  # Rename feature
                transformers.append((new_feature_name, self._get_scaler(scale), [feature]))

        return transformers

    def fit(self, X, y):
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [col for col in self.encoders.keys() if col in X.columns]

        transformers = []

        # Apply multiple scalers to numerical features
        transformers.extend(self._get_scalers(numerical_features))

        # Apply multiple encoders to categorical features
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

    def get_feature_names(self):
        """Return the transformed feature names"""
        if self.pipeline:
            preprocessor = self.pipeline.named_steps["preprocessor"]
            return preprocessor.get_feature_names_out()
        else:
            raise ValueError("Pipeline has not been fitted yet.")

    def get_pipeline(self):
        """Return the underlying scikit-learn pipeline"""
        return self.pipeline


if __name__ == "__main__":
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

    # Define multiple scalings for numerical features
    scalers = {
        "age": "standard+minmax",  # Apply both Standard and MinMax Scaler
        "salary": "robust",        # Apply only Robust Scaler
    }

    # Define multiple encodings for categorical features
    encoders = {
        "city": "onehot+target"  # Apply both OneHot and Target Encoding to "city"
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

    # Print transformed feature names
    print("Final Features Used by Model:")
    print(pipeline.get_feature_names())

    print("Predictions:", predictions)
    print("Transformed Data:\n", pd.DataFrame(transformed_X))

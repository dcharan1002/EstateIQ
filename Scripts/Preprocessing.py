import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class DataPreprocessor:
    def __init__(self, categorical_threshold=10, scaling_method="standard", apply_pca=False, n_components=None):
        """
        Initializes the preprocessor with various configurations.
        - categorical_threshold: Max unique values for one-hot encoding, otherwise label encoding.
        - scaling_method: "standard" for StandardScaler, "minmax" for MinMaxScaler.
        - apply_pca: Boolean to apply PCA.
        - n_components: Number of PCA components (if apply_pca=True).
        """
        self.categorical_threshold = categorical_threshold
        self.scaling_method = scaling_method
        self.apply_pca = apply_pca
        self.n_components = n_components
        self.encoders = {}
        self.imputer = None
        self.scaler = None
        self.pca = None

    def identify_column_types(self, df):
        """Identifies categorical and numerical columns."""
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        return categorical_cols, numerical_cols

    def handle_missing_values(self, df):
        """Fills missing values in both numerical and categorical features."""
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        # Imputing numerical columns with median
        self.imputer = SimpleImputer(strategy="median")
        df[numerical_cols] = self.imputer.fit_transform(df[numerical_cols])

        # Imputing categorical columns with most frequent
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        
        return df

    def encode_categorical(self, df):
        """Encodes categorical features based on their cardinality."""
        categorical_cols, _ = self.identify_column_types(df)
    
        for col in categorical_cols:
            df[col] = df[col].astype(str)  # Convert all categorical values to string
        
            if df[col].nunique() <= self.categorical_threshold:
                # One-hot encoding
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                transformed = encoder.fit_transform(df[[col]])
                new_cols = [f"{col}_{category}" for category in encoder.categories_[0]]
                df[new_cols] = transformed
                df.drop(columns=[col], inplace=True)
                self.encoders[col] = encoder
            else:
                # Label encoding
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
                self.encoders[col] = encoder
    
        return df


    from sklearn.preprocessing import StandardScaler

    def scale_features(self, df):
        """Scales only the numerical features excluding unique identifiers."""
        
        # Identify unique identifier columns (all values are unique)
        unique_id_cols = [col for col in df.columns if df[col].nunique() == len(df)]
        
        # Identify numerical columns that are not unique IDs
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in unique_id_cols]
        
        # Apply scaling only to selected numerical columns
        if self.scaling_method == "standard":
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])  # Scale the numerical columns
            
        return df


    def handle_outliers(self, df):
        """Handles outliers using the IQR method."""
        _, numerical_cols = self.identify_column_types(df)
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        return df

    def apply_dimensionality_reduction(self, df):
        """Applies PCA if configured."""
        if self.apply_pca:
            self.pca = PCA(n_components=self.n_components)
            df = self.pca.fit_transform(df)
        return df

    def preprocess(self, df, target_column=None):
        """Runs the complete preprocessing pipeline."""
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df)
        df = self.encode_categorical(df)
        df = self.scale_features(df)
        
        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            if y.nunique() == 2 or y.dtype == "object":
                X, y = self.balance_classes(X, y)

            X = self.apply_dimensionality_reduction(X)
            return X, y
        else:
            return self.apply_dimensionality_reduction(df)

if __name__ == "__main__":
    # Load raw data
    raw_data = pd.read_csv("Data/Raw/boston_2025.csv")  # Change this to your actual file path

    # Initialize the preprocessor
    preprocessor = DataPreprocessor(apply_pca=False)

    # Define target column (if classification/regression)
    target_column = "TOTAL_VALUE"  # Change this to your dataset's target column

    # Preprocess the data
    X_clean, y_clean = preprocessor.preprocess(raw_data, target_column)

    # Split into train-test
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

    # Save cleaned data
    pd.DataFrame(X_train).to_csv("Data/Processed/X_train_clean.csv", index=False)
    pd.DataFrame(X_test).to_csv("Data/Processed/X_test_clean.csv", index=False)
    pd.DataFrame(y_train).to_csv("Data/Processed/y_train_clean.csv", index=False)
    pd.DataFrame(y_test).to_csv("Data/Processed/y_test_clean.csv", index=False)

    print("Preprocessing completed. Cleaned data saved.")

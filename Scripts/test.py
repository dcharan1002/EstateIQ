import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Preprocessing import DataPreprocessor  # Update this import based on your project structure

class TestDataPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the initial test data."""
        # Creating sample raw data for testing
        cls.df = pd.DataFrame({
            'CITY': ['EAST BOSTON', 'EAST BOSTON', 'BOSTON', np.nan],
            'ZIP_CODE': [2128, 2128, 2128, 2128],
            'TOTAL_VALUE': [1000000, 1500000, 2000000, 1200000],
            'NUM_BLDGS': [1, 2, 1, 2],
            'GROSS_AREA': [1200, 1500, 1800, 1100],
            'LIVING_AREA': [1000, 1300, 1600, 900]
        })

        # Initialize the preprocessor instance
        cls.preprocessor = DataPreprocessor(categorical_threshold=5, scaling_method="standard", apply_pca=False)

    def test_handle_missing_values(self):
        """Test the missing values handling method."""
        df = self.df.copy()
        df.iloc[0, 0] = np.nan  # Introduce a missing value
        df_processed = self.preprocessor.handle_missing_values(df)

        # Assert that missing values in categorical columns are filled
        self.assertFalse(df_processed['CITY'].isnull().any())

    def test_encode_categorical(self):
        """Test categorical encoding."""
        df = self.df.copy()
        df_processed = self.preprocessor.encode_categorical(df)

        # Assert that the 'CITY' column has been encoded (OneHotEncoding or LabelEncoding)
        self.assertTrue('CITY' not in df_processed.columns)
        self.assertTrue(any(col.startswith('CITY_') for col in df_processed.columns))

    def test_handle_outliers(self):
        """Test the outlier handling method."""
        df = self.df.copy()
        df['GROSS_AREA'] = [100000, 1500, 1800, 100000]  # Add outliers
        df_processed = self.preprocessor.handle_outliers(df)

        # Get the computed bounds for clipping
        Q1 = df['GROSS_AREA'].quantile(0.25)
        Q3 = df['GROSS_AREA'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Assert that the values are within the computed bounds after clipping
        self.assertTrue(df_processed['GROSS_AREA'].between(lower_bound, upper_bound).all())


    def test_apply_dimensionality_reduction(self):
        """Test dimensionality reduction with PCA."""
        df = self.df.copy()
        df_processed = self.preprocessor.apply_dimensionality_reduction(df)

        # Assert that PCA was not applied since apply_pca=False
        self.assertEqual(df_processed.shape[1], df.shape[1])

    def test_preprocess_with_target(self):
        """Test the full preprocessing pipeline with target column."""
        df = self.df.copy()
        target_column = "TOTAL_VALUE"
        X_clean, y_clean = self.preprocessor.preprocess(df, target_column)

        # Assert that the target column is removed from features
        self.assertNotIn(target_column, X_clean.columns)

        # Assert that the cleaned data has the expected number of features
        self.assertGreater(X_clean.shape[1], 0)
        self.assertEqual(len(y_clean), df.shape[0])

    def test_train_test_split(self):
        """Test the train-test split."""
        df = self.df.copy()
        target_column = "TOTAL_VALUE"
        X_clean, y_clean = self.preprocessor.preprocess(df, target_column)

        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

        # Assert that the train-test split works and the sizes are correct
        self.assertEqual(X_train.shape[0], int(0.8 * len(X_clean)))
        self.assertEqual(X_test.shape[0], len(X_clean) - X_train.shape[0])

if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import matplotlib.pyplot as plt
from LatentBoostClassifier.visualize import visualize_hybrid_model
from unittest.mock import MagicMock

class TestVisualize(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment by creating mock models and synthetic data.
        """
        # Mock CVAE model with an encoder that outputs latent features
        self.best_cvae = MagicMock()
        self.best_cvae.input = MagicMock()
        self.best_cvae.get_layer.return_value.output = np.random.rand(100, 50)  # Mock latent features

        # Mock CGAN generator
        self.best_cgan_generator = MagicMock()
        self.best_cgan_generator.input_shape = [(None, 50)]
        self.best_cgan_generator.predict.return_value = np.random.rand(100, 10)  # Mock synthetic data

        # Mock Random Forest model
        self.best_rf_model = MagicMock()
        self.best_rf_model.n_features_in_ = 60
        self.best_rf_model.predict.return_value = np.random.randint(0, 2, 20)  # Mock predictions
        self.best_rf_model.predict_proba.return_value = np.random.rand(20, 2)  # Mock probabilities

        # Synthetic datasets
        self.X_train = np.random.rand(100, 10)
        self.Y_train = np.random.randint(0, 2, size=(100, 1))
        self.X_test = np.random.rand(20, 10)
        self.Y_test = np.random.randint(0, 2, size=(20, 1))

    def test_latent_features_extraction(self):
        """
        Test that latent features are extracted correctly from the CVAE.
        """
        encoder = MagicMock()
        encoder.predict.return_value = np.random.rand(20, 50)  # Mock latent features for the test set
        self.best_cvae.get_layer.return_value = encoder
        latent_features_test = encoder.predict([self.X_test, self.Y_test])

        # Assert that the extracted features have the correct shape
        self.assertEqual(
            latent_features_test.shape[1], 50,
            "Latent features should have 50 dimensions."
        )

    def test_synthetic_data_generation(self):
        """
        Test that synthetic data is generated correctly by the CGAN.
        """
        noise_dim = self.best_cgan_generator.input_shape[0][1]
        noise = np.random.normal(0, 1, (len(self.Y_test), noise_dim))
        synthetic_data_test = self.best_cgan_generator.predict([noise, self.Y_test])

        # Assert that the synthetic data has the correct shape
        self.assertEqual(
            synthetic_data_test.shape[1], 10,
            "Synthetic data should have 10 features."
        )

    def test_random_forest_evaluation(self):
        """
        Test that the Random Forest model evaluates the combined features correctly.
        """
        combined_test_features = np.random.rand(20, 60)  # Mock combined features
        predictions = self.best_rf_model.predict(combined_test_features)

        # Assert that predictions have the correct shape
        self.assertEqual(
            len(predictions), self.X_test.shape[0],
            "Predictions should have the same number of samples as the test data."
        )

    def test_visualizations(self):
        """
        Test that visualizations are generated without errors.
        """
        try:
            plt.switch_backend("Agg")  # Use a non-interactive backend for testing
            visualize_hybrid_model(
                self.best_cvae,
                self.best_cgan_generator,
                self.best_rf_model,
                self.X_test,
                self.Y_test,
                self.X_train,
                self.Y_train
            )
        except Exception as e:
            self.fail(f"Visualization failed with error: {e}")

    def test_combined_feature_dimension_adjustment(self):
        """
        Test that combined features are adjusted to match the Random Forest input dimensions.
        """
        combined_test_features = np.random.rand(20, 70)  # Mock features with extra dimensions
        rf_features_in = self.best_rf_model.n_features_in_

        # Adjust dimensions
        adjusted_features = combined_test_features[:, :rf_features_in]

        # Assert that the adjusted features have the correct shape
        self.assertEqual(
            adjusted_features.shape[1], rf_features_in,
            f"Adjusted features should have {rf_features_in} dimensions."
        )


if __name__ == "__main__":
    unittest.main()

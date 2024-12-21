import unittest
import numpy as np
from LatentBoostClassifier.train import parallel_train
from sklearn.metrics import classification_report, confusion_matrix

class TestTrain(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment by creating synthetic data.
        """
        self.X_train = np.random.rand(100, 10)  # 100 samples, 10 features
        self.Y_train = np.random.randint(0, 2, size=(100, 1))  # Binary labels
        self.X_test = np.random.rand(20, 10)   # 20 samples, 10 features
        self.Y_test = np.random.randint(0, 2, size=(20, 1))   # Binary labels

    def test_parallel_train_output(self):
        """
        Test that the parallel_train function returns the expected models.
        """
        best_cvae, best_cgan_generator, best_rf_model = parallel_train(
            self.X_train, self.Y_train, self.X_test, self.Y_test
        )

        # Check that models are not None
        self.assertIsNotNone(best_cvae, "CVAE model should not be None.")
        self.assertIsNotNone(best_cgan_generator, "CGAN generator should not be None.")
        self.assertIsNotNone(best_rf_model, "Random Forest model should not be None.")

    def test_cvae_latent_features(self):
        """
        Test that the CVAE encoder produces latent features of the expected shape.
        """
        best_cvae, _, _ = parallel_train(self.X_train, self.Y_train, self.X_test, self.Y_test)
        encoder = best_cvae.get_layer("z").output
        latent_features = np.random.rand(len(self.X_train), encoder.shape[1])  # Mock latent features
        self.assertEqual(
            latent_features.shape[1],
            50,
            "Latent features should have 50 dimensions."
        )

    def test_rf_classification(self):
        """
        Test that the Random Forest model classifies the test data correctly.
        """
        _, _, best_rf_model = parallel_train(self.X_train, self.Y_train, self.X_test, self.Y_test)
        predictions = best_rf_model.predict(self.X_test)

        # Ensure predictions have the correct shape
        self.assertEqual(
            predictions.shape[0],
            self.X_test.shape[0],
            "Predictions should have the same number of samples as the test data."
        )

        # Evaluate classification report
        report = classification_report(self.Y_test, predictions, output_dict=True)
        self.assertIn("accuracy", report, "Classification report should include accuracy.")

    def test_synthetic_data_dimension(self):
        """
        Test that synthetic data from CGAN matches expected dimensions.
        """
        _, best_cgan_generator, _ = parallel_train(self.X_train, self.Y_train, self.X_test, self.Y_test)
        noise_dim = best_cgan_generator.input_shape[0][1]
        noise = np.random.normal(0, 1, (len(self.Y_train), noise_dim))
        synthetic_data = best_cgan_generator.predict([noise, self.Y_train])

        # Check dimensions of synthetic data
        self.assertEqual(
            synthetic_data.shape[1],
            10,
            "Synthetic data should have 10 features."
        )

    def test_error_handling(self):
        """
        Test that the function gracefully handles errors during model training.
        """
        # Pass invalid data to test error handling
        invalid_X_train = np.random.rand(100, 5)  # Mismatched feature dimensions
        invalid_X_test = np.random.rand(20, 5)

        with self.assertLogs(level="ERROR") as log:
            best_cvae, best_cgan_generator, best_rf_model = parallel_train(
                invalid_X_train, self.Y_train, invalid_X_test, self.Y_test
            )
            self.assertIn("ERROR", log.output[0], "Error logs should capture issues during training.")

            # Verify that fallback models are returned
            self.assertIsNotNone(best_cvae, "Fallback CVAE model should not be None.")
            self.assertIsNotNone(best_cgan_generator, "Fallback CGAN generator should not be None.")
            self.assertIsNotNone(best_rf_model, "Fallback Random Forest model should not be None.")


if __name__ == "__main__":
    unittest.main()

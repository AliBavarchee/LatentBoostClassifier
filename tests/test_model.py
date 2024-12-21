import unittest
import numpy as np
from tensorflow.keras import layers, models
from LatentBoostClassifier.model import (
    build_cvae,
    build_cgan,
    tune_cvae,
    tune_cgan,
    tune_random_forest,
)
from unittest.mock import MagicMock, patch
from sklearn.ensemble import RandomForestClassifier

class TestModel(unittest.TestCase):
    def setUp(self):
        """
        Set up mock data and shared resources for testing.
        """
        self.X_train = np.random.rand(100, 10)  # 100 samples, 10 features
        self.Y_train = np.random.randint(0, 2, size=(100, 1))  # Binary labels
        self.X_test = np.random.rand(20, 10)   # 20 samples, 10 features
        self.Y_test = np.random.randint(0, 2, size=(20, 1))   # Binary labels
        self.shared_dict = {}

    def test_build_cvae(self):
        """
        Test that the CVAE model is built correctly and has expected layers.
        """
        hp_mock = MagicMock()
        hp_mock.Int.side_effect = lambda name, min_value, max_value, step: 50 if name == "latent_dim" else 128
        hp_mock.Choice.side_effect = lambda name, values: 1e-4

        cvae = build_cvae(hp_mock)

        self.assertIsInstance(cvae, models.Model, "CVAE should be an instance of tf.keras.Model.")
        self.assertEqual(cvae.input_shape, [(None, 10), (None, 1)], "CVAE input shapes should match [(None, 10), (None, 1)].")
        self.assertIn("custom_loss_layer", [layer.name for layer in cvae.layers], "CVAE should include the custom loss layer.")

    def test_build_cgan(self):
        """
        Test that the CGAN model is built correctly and includes generator and discriminator.
        """
        hp_mock = MagicMock()
        hp_mock.Int.side_effect = lambda name, min_value, max_value, step: 50 if name == "latent_dim" else 128
        hp_mock.Choice.side_effect = lambda name, values: 1e-4

        cgan = build_cgan(hp_mock)

        self.assertIsInstance(cgan, models.Model, "CGAN should be an instance of tf.keras.Model.")
        self.assertIn("generator", [layer.name for layer in cgan.layers], "CGAN should include the generator layer.")
        self.assertIn("discriminator", [layer.name for layer in cgan.layers], "CGAN should include the discriminator layer.")

    @patch("LatentBoostClassifier.model.GridSearchCV")
    def test_tune_random_forest(self, mock_grid_search):
        """
        Test that the Random Forest tuner fits correctly and stores the best model.
        """
        latent_features = np.random.rand(100, 50)  # Mock latent features
        synthetic_data = np.random.rand(100, 50)   # Mock synthetic data
        mock_best_model = RandomForestClassifier()
        mock_grid_search.return_value.best_estimator_ = mock_best_model

        tune_random_forest(latent_features, synthetic_data, self.Y_train, self.shared_dict)

        self.assertIn("rf_model", self.shared_dict, "Random Forest model should be stored in shared_dict.")
        self.assertIsInstance(
            self.shared_dict["rf_model"], RandomForestClassifier, "Stored model should be a RandomForestClassifier."
        )

    @patch("LatentBoostClassifier.model.kt.Hyperband")
    def test_tune_cvae(self, mock_hyperband):
        """
        Test that the CVAE tuner runs correctly and updates shared_dict.
        """
        mock_best_model = MagicMock()
        mock_hyperband.return_value.get_best_models.return_value = [mock_best_model]

        tune_cvae(self.X_train, self.Y_train, self.X_test, self.Y_test, self.shared_dict)

        self.assertIn("best_cvae", self.shared_dict, "Best CVAE model should be stored in shared_dict.")
        self.assertEqual(self.shared_dict["best_cvae"], mock_best_model, "Stored model should match the best CVAE model.")

    @patch("LatentBoostClassifier.model.kt.Hyperband")
    def test_tune_cgan(self, mock_hyperband):
        """
        Test that the CGAN tuner runs correctly and updates shared_dict.
        """
        mock_best_model = MagicMock()
        mock_best_model.get_layer.return_value = MagicMock()
        mock_hyperband.return_value.get_best_models.return_value = [mock_best_model]

        tune_cgan(self.X_train, self.Y_train, self.X_test, self.Y_test, self.shared_dict)

        self.assertIn("best_cgan_generator", self.shared_dict, "Best CGAN generator should be stored in shared_dict.")
        self.assertEqual(
            self.shared_dict["best_cgan_generator"],
            mock_best_model.get_layer.return_value,
            "Stored generator should match the best CGAN generator layer."
        )


if __name__ == "__main__":
    unittest.main()

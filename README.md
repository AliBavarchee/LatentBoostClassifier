__________________________________________________<p align="Center">![logo](https://github.com/AliBavarchee/LatentBoostClassifier/blob/main/HybridGenerativeModel_Slogo.png)</p>__________________________________________________
---
__     ___  ______  ____ __  __ ______
       ||    // \\ | || | ||    ||\ || | || |
       ||    ||=||   ||   ||==  ||\\||   ||  
       ||__| || ||   ||   ||___ || \||   ||    
	   ____    ___     ___    __  ______
          || ))  // \\   // \\  (( \ | || |
          ||=)  ((   )) ((   ))  \\    ||  
          ||_))  \\_//   \\_//  \_))   ||
  	        __                         
           /   |  _   _  _ . (_ .  _  _ 
           \__ | (_| _) _) | |  | (- |
             __       ___        ____   
            /_ |     |__ \      |___ \  
             | |        ) |       __) | 
             | |       / /       |__ <  
             | |  __  / /_   __  ___) | 
             |_| (__)|____| (__)|____/
---
---
# LatentBoostClassifier: Hybrid Generative Model: CVAE + CGAN + RF
---
---
| http://dx.doi.org/10.13140/RG.2.2.11351.79522 |
---

This repository implements a **hybrid generative model+** combining a Conditional Variational Autoencoder (CVAE), a Conditional Generative Adversarial Network (CGAN), and a Random Forest Classifier (RFC). The hybrid model is designed for complex classification tasks by leveraging latent feature extraction (CVAE), synthetic data generation (CGAN), and robust classification (RFC).

---

## Features

1. **Conditional Variational Autoencoder (CVAE)**:
   - Learns a latent representation of input data.
   - Uses a custom loss function combining reconstruction loss and KL divergence.

2. **Conditional Generative Adversarial Network (CGAN)**:
   - Generates high-quality synthetic data conditioned on class labels.

3. **Random Forest Classifier**:
   - Trained on a combination of real and synthetic features to enhance classification performance.

4. **Parallel Model Training**:
   - The CVAE and CGAN models are trained concurrently using Python's `multiprocessing` module.

5. **Hyperparameter Tuning**:
   - Utilizes **Keras Tuner** for optimizing the CVAE and CGAN hyperparameters.
   - Random Forest is tuned using **GridSearchCV**.

---

## Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.8+
- Scikit-learn
- Keras Tuner
- Matplotlib
- Seaborn
- Multiprocessing

### Install Required Packages
```bash
pip install tensorflow keras-tuner scikit-learn matplotlib seaborn
```

---

---

**Install the Package**

the package can be installed the package via PyPI:
```bash
pip install LatentBoostClassifier
```

Alternatively, install directly from GitHub:
```bash
pip install git+https://github.com/AliBavarchee/LatentBoostClassifier.git
```

---


## Usage

---

### Using the Package

After installation, the package can be imported and used like this:
```python
from LatentBoostClassifier import parallel_train, visualize_hybrid_model

# Train the hybrid model
best_cvae, best_cgan_generator, best_rf_model = parallel_train(X_train, Y_train, X_test, Y_test)

# Visualize results
visualize_hybrid_model(best_cvae, best_cgan_generator, best_rf_model, X_test, Y_test, X_train, Y_train)
```

---

### Training the Hybrid Model
Train the CVAE, CGAN, and Random Forest models using your dataset:
```python
# Import the training function
from LatentBoostClassifier import parallel_train

# Define training and testing datasets
X_train, Y_train = ...  # Load or preprocess your training data
X_test, Y_test = ...    # Load or preprocess your test data

# Train the hybrid model
best_cvae, best_cgan_generator, best_rf_model = parallel_train(X_train, Y_train, X_test, Y_test)
```

---

### Visualizing Results
Evaluate and visualize the performance of the hybrid model:
```python
from LatentBoostClassifier import visualize_hybrid_model

# Visualize results
visualize_hybrid_model(best_cvae, best_cgan_generator, best_rf_model, X_test, Y_test, X_train, Y_train)
```

---

## Code Overview

### 1. CVAE
- **Purpose**: Extract latent features from input data.
- **Key Components**:
  - Custom loss function: Combines reconstruction loss and KL divergence.
  - Encoder-Decoder architecture.
- **Hyperparameter Tuning**:
  - Latent dimensions.
  - Dense layer units.
  - Learning rate.

### 2. CGAN
- **Purpose**: Generate synthetic data conditioned on class labels.
- **Key Components**:
  - Generator: Produces synthetic samples.
  - Discriminator: Distinguishes real and synthetic samples.
- **Hyperparameter Tuning**:
  - Latent dimensions.
  - Dense layer units.
  - Learning rate.

### 3. Random Forest
- **Purpose**: Classify combined real and synthetic data.
- **Key Components**:
  - Trained on latent features (CVAE) and synthetic data (CGAN).
  - Grid search for hyperparameter optimization.

---

## Visualization

### Classification Performance
- **Confusion Matrix**:
  Visualize actual vs. predicted class distributions.
- **Classification Report**:
  Display precision, recall, F1-score, and accuracy metrics.

### Latent Space
- Use **t-SNE** to visualize the CVAE's latent space.

### Synthetic Data
- Compare real and synthetic data distributions using KDE plots.

### ROC Curve
- Evaluate the Random Forest classifier with an ROC curve and compute AUC.

---

## Key Functions

### Model Training
```python
parallel_train(X_train, Y_train, X_test, Y_test)
```
- Trains the CVAE, CGAN, and Random Forest models in parallel.

### Visualization
```python
visualize_hybrid_model(best_cvae, best_cgan_generator, best_rf_model, X_test, Y_test, X_train, Y_train)
```
- Visualizes the hybrid model's results, including latent space and classification performance.

---

## Examples

### Training Example
```python
# Example Dataset
X_train = np.random.rand(80, 10)
Y_train = np.random.randint(0, 2, size=(80, 1))
X_test = np.random.rand(20, 10)
Y_test = np.random.randint(0, 2, size=(20, 1))

# Train the hybrid model
best_cvae, best_cgan_generator, best_rf_model = parallel_train(X_train, Y_train, X_test, Y_test)
```

### Visualization Example
```python
# Visualize results
visualize_hybrid_model(best_cvae, best_cgan_generator, best_rf_model, X_test, Y_test, X_train, Y_train)
```

---

## Contributions

Feel free to fork, improve, and submit a pull request! For major changes, please open an issue first to discuss what you'd like to change.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
---

=============================================<p align="Center">![ALI BAVARCHIEE](https://github.com/AliBavarchee/AliBavarchee/blob/main/logo.png)</p>=============================================
=====
| https://github.com/AliBavarchee/ |
----
| https://www.linkedin.com/in/ali-bavarchee-qip/ |
----

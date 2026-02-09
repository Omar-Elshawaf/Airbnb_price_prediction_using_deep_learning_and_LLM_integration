# Airbnb Price Prediction and LLM Interpretation

## Overview

This project uses a Deep Neural Network (DNN) to predict the price of Airbnb listings based on their features. The project also integrates a **Large Language Model (LLM)** to provide interpretability for the model by generating human-readable justifications for the price predictions.

The goal is to predict Airbnb prices accurately, while also bridging the gap between quantitative prediction (price) and qualitative interpretability (reasons for pricing).

### Key Features:
- **Deep Neural Network** for price prediction (regression).
- **LLM (Flan-T5)** for few-shot classification and reasoning (interpretability).
- **Data Preprocessing** for missing data handling, normalization, and feature engineering.
- **Model Evaluation** with performance metrics like **R² score**, **MAE**, and **classification accuracy**.

---

## Code Overview

### Key Cells:
  1. **Data Preprocessing**: The dataset is split into training, validation, and testing sets. Missing data is handled, and numeric features are normalized using Z-score normalization.
  2. **Neural Network Training**: The architecture uses a multilayer feedforward neural network with dropout and batch normalization to prevent overfitting.
  3. **LLM Integration**: The **Flan-T5** model is used for few-shot classification of price tiers (Low, Medium, High) and generating textual explanations for predictions based on listing features.


---

## Methods and Algorithms

### Data Preprocessing and Feature Engineering
- **Data Splitting**: The dataset is split into:
  - **Training (70%)**
  - **Validation (15%)**
  - **Testing (15%)**
  
- **Handling Missing Data**: Numerical features (e.g., bathrooms, review scores) were imputed using the median, and categorical features (e.g., room type, city) were imputed using the most frequent value.

- **Normalization**: The features were normalized using **Z-score normalization** to ensure all inputs have similar scales, which helps the model train more effectively.

- **Feature Correlation Analysis**: Feature correlations were analyzed to understand relationships between features and the target variable (price).

### Neural Network Architecture (Regression)
- The neural network is a **Multilayer Feedforward Neural Network** using **TensorFlow/Keras**:
  - **Input Layer**: Matches the dimensionality of the preprocessed feature vector (one-hot encoded categoricals + scaled numerics).
  - **Hidden Layers**: 
    - Dense Layer (512 units, ReLU)
    - Batch Normalization
    - Dropout (0.3) to prevent overfitting
    - Dense Layer (256 units, ReLU) + BN + Dropout (0.25)
    - Dense Layer (128 units, ReLU) + BN + Dropout (0.20)
  - **Output Layer**: A single neuron with **Linear activation** to predict the continuous price value.
  - **Loss Function**: **Huber Loss**, which is less sensitive to outliers compared to **Mean Squared Error (MSE)**.

### LLM (Flan-T5) Integration
- **Few-Shot Classification**: The **Flan-T5** model is used for classifying price tiers (Low, Medium, High) using few-shot learning. It was given example feature vectors and corresponding labels to predict unseen data.
- **Interpretability (LLM Reasoning)**: The LLM was prompted to act as a "Pricing Analyst" to generate textual explanations for why a property is priced at a certain level based on its features.

---

## Experimental Results

### Neural Network Performance
- **Training Loss**: The Huber loss decreased significantly from ~4993 in Epoch 1 to ~1441 in Epoch 12, indicating successful learning.
- **Validation Loss**: The validation loss stabilized around 1678, suggesting effective regularization.

### Regression Accuracy
- **R² Score**: 0.594, which indicates that the model explains approximately **60%** of the variance in the pricing.
- **MAE (Mean Absolute Error)**: $48.61, meaning the predictions are off by an average of $48.61.
- **MAPE**: 31.49%.
- **Accuracy (within 20%)**: **45.23%** of predictions were within 20% of the actual price.

### Classification Performance
- **Neural Network Classification**: The NN achieved **72.5%** classification accuracy in predicting the price categories (Low, Medium, High).
- **LLM Classification**: The LLM achieved **27.5%** accuracy, struggling significantly with the Medium category.

### LLM Reasoning
- The LLM excelled in generating human-readable explanations for pricing, such as:
  - **"Strict cancellation policy"**
  - **"Real bed"**
  - **Location-based insights (e.g., "Duboce Triangle")**

---

## Results Visualization
The project includes several visualizations:
- **Predicted vs. True Price**: A plot comparing predicted prices to actual prices.
- **Confusion Matrix**: For both the neural network and LLM classification results.
- **LLM Reasoning Examples**: Text-based justifications for price predictions.
- **Training and Validation Loss**: Visualizations of the loss reduction over training epochs.
<img width="1060" height="986" alt="image" src="https://github.com/user-attachments/assets/2c943ed7-3a08-4e24-961b-c5e604b6d31b" />
<img width="1035" height="402" alt="image" src="https://github.com/user-attachments/assets/2d4f2c9d-343e-468f-8eeb-2fb3e269ad97" />
<img width="549" height="479" alt="image" src="https://github.com/user-attachments/assets/7c165d11-de57-47c0-9610-d233659ca348" />
<img width="1010" height="378" alt="image" src="https://github.com/user-attachments/assets/60c65ca0-e637-4996-8351-03a8f9606969" />
<img width="1025" height="402" alt="image" src="https://github.com/user-attachments/assets/cc65aff5-537d-4a2c-bf1e-1d2245ef06ee" />
<img width="982" height="402" alt="image" src="https://github.com/user-attachments/assets/de360d63-5be8-4e51-89c2-299b4ba37c2a" />
<img width="560" height="556" alt="image" src="https://github.com/user-attachments/assets/59b02508-eba2-45f6-8c37-55d0a43c4215" />
<img width="537" height="479" alt="image" src="https://github.com/user-attachments/assets/96767116-7a87-4daf-83aa-970aded3075b" />
<img width="518" height="479" alt="image" src="https://github.com/user-attachments/assets/7066f896-001b-48f4-801d-7d32975e50ef" />
<img width="1336" height="384" alt="image" src="https://github.com/user-attachments/assets/66b8ffbe-b9c8-4853-97f7-5a61103b94db" />

---


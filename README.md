# E-commerce Return Risk Prediction Using Logistic Regression

## Project Overview
This project predicts the risk of e-commerce product returns using a Logistic Regression model. It uses data preprocessing and trained model components to estimate return risk through a Streamlit web application.

## Objectives
- Analyze e-commerce return data.
- Preprocess data for machine learning.
- Encode categorical features and scale numerical features where required.
- Predict return risk using Logistic Regression.
- Provide an interactive interface using a Streamlit web application.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Project Files
| File | Description |
|---|---|
| `app.py` | Streamlit web application |
| `E-commerce_Return.ipynb` | Model development and analysis |
| `e-commerce_return.csv` | Dataset |
| `bias.pkl` | Saved model bias |
| `encoder.pkl` | Saved categorical encoder |
| `scaler.pkl` | Saved feature scaler |
| `weight.npy` | Saved model weights |
| `requirements.txt` | Python dependencies |

## Project Workflow
1. Load and explore the dataset.
2. Clean and preprocess the data.
3. Encode categorical variables and scale features as needed.
4. Train the Logistic Regression model.
5. Save the model parameters and preprocessing objects.
6. Use the saved components in the Streamlit application.
7. Predict return risk for user-provided inputs.

## Installation and Usage

### 1. Clone the repository
Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd E-commerce-Return-Risk-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

## Model Evaluation
Evaluate the model using appropriate classification metrics, such as accuracy, precision, recall, F1-score, and a confusion matrix. Add the actual results after verifying them in the notebook.

## Disclaimer
Predictions are estimates based on the training data and should not be treated as guaranteed outcomes.

## Author
Deepika Panneer Selvam

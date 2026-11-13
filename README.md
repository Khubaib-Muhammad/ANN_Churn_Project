# 🏦 Bank Customer Churn Prediction using Artificial Neural Networks (ANN)

An end-to-end Deep Learning project that predicts whether a bank customer is likely to leave the bank using an Artificial Neural Network (ANN). The project includes data preprocessing, model training, evaluation, and an interactive Streamlit web application for real-time predictions.

## 🚀 Live Demo

**[Launch Bank Customer Churn Predictor](https://nxrg5gffhmxts2uo3sxw6s.streamlit.app/)**

## 📌 Project Overview

Customer churn occurs when customers stop using a company's services. For banks, identifying customers who are likely to leave is important for improving customer retention and developing effective business strategies.

This project uses an Artificial Neural Network trained on the Bank Customer Churn dataset to predict customer churn based on demographic, financial, and account-related information.

The model predicts:

- **0 – Stay:** Customer is likely to remain with the bank.
- **1 – Leave:** Customer is likely to leave the bank.

## 🎯 Objectives

- Explore and understand customer churn data.
- Perform data cleaning and preprocessing.
- Apply One-Hot Encoding to categorical features.
- Apply StandardScaler to numerical features.
- Build and train an Artificial Neural Network.
- Evaluate model performance using classification metrics.
- Develop an interactive web application using Streamlit.
- Generate churn probabilities for individual customers.

## 🧠 ANN Architecture

The project uses a Sequential Artificial Neural Network with the following architecture:

| Layer | Neurons | Activation |
|---|---:|---|
| Input Layer | 13 | None |
| Hidden Layer 1 | 16 | ReLU |
| Hidden Layer 2 | 8 | ReLU |
| Output Layer | 1 | Sigmoid |

**Model Configuration:**

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Evaluation Metric: Accuracy
- Epochs: 50
- Batch Size: 32
- Validation Split: 20%
- Early Stopping: Patience of 5 epochs
- Decision Threshold: 0.5

## 📊 Dataset

The project uses the **Bank Customer Churn Modelling Dataset**, containing 10,000 customer records and 14 original columns.

### Features Used

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Customer's country |
| Gender | Customer's gender |
| Age | Customer's age |
| Tenure | Years with the bank |
| Balance | Bank account balance |
| NumOfProducts | Number of bank products |
| HasCrCard | Credit card ownership |
| IsActiveMember | Active membership status |
| EstimatedSalary | Estimated annual salary |
| Exited | Target variable |

The following columns are removed because they are identifiers rather than useful prediction features:

- RowNumber
- CustomerId
- Surname

After preprocessing, the 10 input features are transformed into **13 numerical features**.

## ⚙️ Technologies Used

- **Python** – Programming language
- **TensorFlow / Keras** – Artificial Neural Network development
- **Pandas & NumPy** – Data manipulation
- **Scikit-learn** – Data preprocessing and evaluation
- **Matplotlib** – Data visualization and training loss plots
- **Joblib** – Saving and loading the preprocessor
- **Streamlit** – Interactive web application

## 📁 Project Structure

```text
ANN_Churn_Project/
│
├── ANN_Churn_Project.ipynb
├── Churn_Modelling.csv
├── app.py
├── model.keras
├── preprocessor.pkl
├── requirements.txt
└── README.md
```

## 🛠️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Khubaib-Muhammad/ANN_Churn_Project.git
```

### 2. Navigate to the project directory

```bash
cd ANN_Churn_Project
```

### 3. Create a virtual environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 🌐 Web Application Features

The Streamlit application provides:

- Customer information input fields.
- Dropdown menus for categorical features.
- Numerical inputs for financial and demographic information.
- One-click churn prediction.
- Churn probability displayed as a percentage.
- Clear prediction verdict:
  - Likely to STAY
  - Likely to LEAVE

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-Score
- Classification Report
- Training and Validation Loss Curves

These metrics help assess how effectively the ANN identifies customers who are likely to leave.

## 💾 Model Artifacts

Two important files are saved after training:

- `model.keras` – Trained Artificial Neural Network.
- `preprocessor.pkl` – Fitted preprocessing pipeline containing encoding and scaling transformations.

Both files are required by the Streamlit application to generate predictions for new customers.

## 🔮 Future Improvements

- Experiment with deeper ANN architectures.
- Improve recall for the churn class.
- Experiment with class weights to handle class imbalance.
- Tune the prediction threshold.
- Compare ANN performance with traditional Machine Learning algorithms.
- Improve model interpretability using SHAP.
- Experiment with additional hyperparameter optimization techniques.

## 👨‍💻 Author

**Khubaib Muhammad**

- GitHub: [@Khubaib-Muhammad](https://github.com/Khubaib-Muhammad)
- Project Repository: [ANN_Churn_Project](https://github.com/Khubaib-Muhammad/ANN_Churn_Project)
- Live Application: [Streamlit Demo](https://nxrg5gffhmxts2uo3sxw6s.streamlit.app/)

---

⭐ If you find this project interesting, consider giving the repository a star!

*This project was developed as part of a Deep Learning / Artificial Neural Networks mini-project.*

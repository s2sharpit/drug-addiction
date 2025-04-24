
# Binary Classification: Feature Selection & Model Comparison

This project focuses on feature selection and evaluating multiple machine learning models for a binary classification problem. The pipeline includes correlation analysis, multiple feature selection techniques, model training, evaluation, and comparison of various classifiers including Logistic Regression, Random Forest, SVM, XGBoost, and KNN.

---

## 📁 Project Structure

```
├── feature_selection.ipynb         # Notebook for feature correlation and feature selection
├── model_training.ipynb            # Notebook for model training and evaluation
├── train_drop_records.csv          # Original dataset
├── selected_features.csv           # Dataset with selected features after feature selection
├── target.csv                      # Target values
├── correlation_matrix.png          # Heatmap of feature correlations
├── feature_selection_heatmap.png   # Normalized feature selection scores
├── model_comparison.png            # Comparison plot of Accuracy and AUC for models
├── README.md                       # Project documentation
```

---

## 📌 Features

- Drops highly correlated features (threshold: 0.9)
- Selects top features using:
  - Chi-Square
  - Mutual Information
  - Random Forest feature importance
  - Recursive Feature Elimination (RFE)
- Trains and evaluates:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
  - K-Nearest Neighbors (KNN)
- Saves and plots:
  - Feature selection scores heatmap
  - Model comparison (Accuracy & AUC)
- Saves the best-performing model

---

## 🚀 Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create and Activate a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. **Install Requirements**

   ```bash
   pip install -r requirements.txt
   ```

   If a `requirements.txt` is not available, install manually:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
   ```

4. **Run the Notebooks**

   Open both notebooks in Jupyter or VSCode:

   - `feature_selection.ipynb`
   - `model_training.ipynb`

---

## ✅ Results

- Top features are selected based on normalized scores across four selection methods.
- All models are evaluated based on **Accuracy** and **AUC**.
- The best model (based on AUC) is saved as a `.pkl` file using `joblib`.

---

## 📊 Output Visuals

- **correlation_matrix.png** – Heatmap showing feature correlation
- **feature_selection_heatmap.png** – Combined view of feature importance across methods
- **model_comparison.png** – Accuracy and AUC comparison of trained models

---

## 🧠 Models Used

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- K-Nearest Neighbors (KNN)

---

## 📦 Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib

---

## ✨ Author

Created by [Your Name]. Feel free to reach out for collaboration or questions!

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

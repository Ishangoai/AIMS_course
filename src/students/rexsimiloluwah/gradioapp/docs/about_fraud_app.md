## About This Application

This Credit Card Fraud Detection app uses machine learning to identify potentially fraudulent transactions
based on various transaction features.

---

<details>
<summary><b>⚙️ How It Works</b></summary>

The model analyzes 12 different features of a credit card transaction:

- **V3, V4, V7, V9, V10, V11, V12, V14, V16, V17, V18** – PCA-transformed features derived from
  the original transaction data (confidential due to privacy).  
- **Amount** – The transaction amount in dollars.

</details>

<details>
<summary><b>🧠 Model Information</b></summary>

- **Algorithm**: `RandomForestClassifier`  
- The model was trained on a highly imbalanced dataset containing both normal and fraudulent transactions.  
- **Feature Selection**: RandomForest feature importance scores were used to identify and retain the most relevant
  12 features for training.  
- The trained model achieved strong performance with high accuracy and balanced F1-score, demonstrating
  robust discrimination between fraudulent and legitimate transactions.

</details>

<details>
<summary><b>🧰 Tools Used</b></summary>

- **Dagster** – for orchestration and pipeline management  
- **MLflow** – for model tracking, versioning, and deployment  
- **Gradio** – for interactive model inference and visualization  

</details>


<details>
<summary><b>💡 Using the Application</b></summary>

1. **Manual Input** – Adjust feature sliders manually.  
2. **Sample Cases** – Load a “Normal” or “Fraud” transaction example.  
3. **Predict** – Click “🔍 Predict” to analyze the transaction.  
4. **Clear** – Reset all values to zero.

</details>


<details>
<summary><b>🧾 Interpretation Guide</b></summary>

- ✅ **Low Risk Transaction** – Predicted as legitimate.  
- ⚠️ **High Risk Transaction** – Flagged as potentially fraudulent.  
- ❌ **Prediction Failed** – An error occurred during inference.

</details>


<details>
<summary><b>📘 Note</b></summary>

This is an educational project demonstrating machine learning applications in fraud detection.  
In real-world systems, fraud detection models use additional behavioral, geographic, and device-based features
combined with real-time anomaly detection for improved accuracy.

</details>

---

**Developed by**: Khadija Edarzi & Similoluwa Okunowo  
**Course**: MLOps Course at *AIMS (African Institute for Mathematical Sciences)*  
**Credits**: Olivier, Vincent, Cyrille, Jan, and the AIMS MLOps course team from Ishango.ai for the amazing course content ❤️!
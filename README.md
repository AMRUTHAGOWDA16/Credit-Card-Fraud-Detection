# Credit Card Fraud Detection Using Machine Learning & Flask

## 📌 Project Overview
This project detects fraudulent credit card transactions using Machine Learning.  
A trained classification model predicts whether a transaction is **Fraud** or **Genuine**.  

It includes:
- Machine Learning model (trained on Kaggle dataset)
- Flask Web App interface
- Secure login page
- Model training page
- Model testing page
- Transaction analysis and results page

---

## 📂 Project Folder Structure
```
credit-card-fraud-detection/
│── app.py                # Flask web application
│── train_model.py        # Machine learning model training script
│── requirements.txt      # Libraries needed
│── README.md             # Project documentation
│
├── data/
│   └── creditcard.csv    # Kaggle fraud dataset
│
├── models/
│   ├── model.joblib
│   ├── scaler.joblib
│   └── metrics.joblib
│
├── templates/            # HTML files
└── static/               # CSS files
```

---

## ⚙️ Tools & Technologies
| Category | Tools |
|--------|-----------|
Language | Python  
Web Framework | Flask  
ML Libraries | Scikit-Learn, Pandas, NumPy  
Frontend | HTML + CSS  
Dataset | Credit Card Fraud Dataset (Kaggle)

---

## 📊 Dataset Info
The dataset contains real credit card transaction records.  
Fraud cases are very rare (highly imbalanced dataset).

Source: Kaggle — Credit Card Fraud Detection Dataset

---

## 🚀 How to Run the Project

### **Step-1: Install Dependencies**
```
pip install -r requirements.txt
```

### **Step-2: Train Model**
```
python train_model.py
```

### **Step-3: Run Flask App**
```
python app.py
```

### **Step-4: Open in Browser**
```
http://127.0.0.1:5000/
```

---

## 🔥 Features
✅ User Login Authentication  
✅ Train Model Button  
✅ Predict Fraud Transactions  
✅ Upload/Test CSV file  
✅ Shows Prediction Result  
✅ Simple web UI  

---

## 🧠 Project Outcome
This project shows how machine learning can be used to detect fraud transactions in banking and cybersecurity systems.

---


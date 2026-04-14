# 🚗 CAN Bus Intrusion Detection System (IDS)

An AI-based Intrusion Detection System designed to detect and classify malicious or abnormal messages in CAN (Controller Area Network) bus data used in automotive systems.

---

## 📌 Project Overview

Modern vehicles rely heavily on CAN bus communication between Electronic Control Units (ECUs). However, this makes them vulnerable to attacks such as spoofing, denial-of-service, and injection attacks.

This project uses machine learning / deep learning techniques to analyze CAN messages and classify them as:

- ✅ Normal traffic  
- ⚠️ Attack / Intrusion traffic  

---

## 🎯 Objectives

- Detect abnormal CAN bus messages in real-time or offline logs  
- Improve vehicle cybersecurity using AI-based classification  
- Build a scalable intrusion detection pipeline  

---

## 🧠 Methodology

1. Data Collection from CAN logs / datasets  
2. Preprocessing (cleaning, encoding, scaling)  
3. Feature extraction from CAN IDs and payload  
4. Model training (e.g., LSTM / CNN / Random Forest / etc.)  
5. Evaluation using accuracy, precision, recall, F1-score  
6. Prediction on new CAN messages  

---

## ⚙️ Installation
```bash
git clone https://github.com/sanjeev-kumar-04/CAN-Bus-Intrusion-Detection-System.git
cd CAN-Bus-Intrusion-Detection-System

pip install -r requirements.txt
```
## 🚀 How to Run

### ▶️ Train Model
```bash
python main.py
```

### ▶️ Run Simulation
```bash
python test.py
```

### ▶️ Run Web App (Streamlit)
```bash
streamlit run app.py
```

---

## 🧪 Dataset

- CAN bus log dataset from: https://ocslab.hksecurity.net/Datasets/car-hacking-dataset 
- Includes both **normal** and **attack traffic samples**  
- Used for training and evaluation of the intrusion detection model  

---

## 🔐 Attack Types Detected

- Denial of Service (DoS)  
- Spoofing Attacks  
- Injection Attacks  
- Fuzzy Attacks  

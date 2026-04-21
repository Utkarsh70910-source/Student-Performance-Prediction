# 🎓 Student Performance Prediction System

A Machine Learning based web application that predicts a student's final exam score (G3) based on academic performance and lifestyle factors.

---

## 🚀 Features

- Predict final score (G3) out of 100
- Uses Machine Learning models
- Based on student study habits and academic data
- Simple and user-friendly web interface


---

## 📂 Project Structure
End-to-End-Student-Performance-Prediction/
│
├── data/
│ └── stud.csv
│
├── src/
│ ├── components/
│ ├── pipelines/
│ ├── utils/
│ └── logger.py
│
├── templates/
│ └── index.html
│
├── Artifacts/
│ ├── model.pkl
│ └── preprocessor.pkl
│
├── app.py
├── requirements.txt
└── README.md

---

## ⚙️ Setup & Installation

### Clone Project

git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git  
cd YOUR_REPO  

---

### Create Virtual Environment

python -m venv venv  

---

### Activate Environment

Windows:  
venv\Scripts\activate  

Mac/Linux:  
source venv/bin/activate  

---

### Install Dependencies

pip install -r requirements.txt  


---

## ▶️ Run the Project

### Train the Model

python src/pipelines/Training_pipeline.py  

---

### Start the Application

python app.py  

---

### Open in Browser

http://127.0.0.1:5000  
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









---

## 📊 Input Features

| Feature | Description |
|--------|------------|
| G1 | First internal exam (0–20) |
| G2 | Second internal exam (0–20) |
| Study Time | Daily study duration |
| Absences | Number of missed classes |
| Failures | Past failures |
| Sleep Hours | Daily sleep time |
| Social Media | Usage per day |
| Medu | Mother education level |
| Fedu | Father education level |
| Famrel | Family relationship (1–5) |
| Internet | Internet access (Yes/No) |
| Higher | Interest in higher education |

---

## 🎯 Output

- Predicted Final Score (G3) out of 100



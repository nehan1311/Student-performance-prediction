# Student Performance Project

A machine learning project for predicting student performance based on their habits.

## Project Structure

```
student-performance-project/
├── backend/
│   ├── app.py
│   ├── model_training.py
│   ├── models/
│   │   ├── classifier.pkl
│   │   ├── regressor.pkl
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
├── data/
│   └── student_habits_performance.csv
├── README.md
└── requirements.txt
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the models:
```bash
python backend/model_training.py
```

3. Run the application:
```bash
python backend/app.py
```


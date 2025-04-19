END TO END ML PROJECT

# Student Math Score Predictor

End-to-end ML pipeline to predict students’ math scores, deployed as a Flask API.

## Features
- **Data Ingestion** from CSV (and extendable to SQL/S3)
- **Data Transformation** with `ColumnTransformer` (OneHot + StandardScaler)
- **Model Training & Tuning** (multiple regressors + GridSearch)
- **Deployment** via Flask with two endpoints and HTML form
- **Logging** of requests & predictions in SQLite

## 📁 Project Structure


## 📦 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/Kolawoler/Student_ML_Project.git
   cd your-repo

2 ## **Create and activate a virtual environment:**
   python3 -m venv venv
   source venv/bin/activate

 3. ## pip install -r requirements.txt
      pip install -r requirements.txt

**Usage**
Ingest & split data:
python src/ingestion.py

**Transform features:**
python src/transformation.py

**Train & tune models:**
python src/training.py

Run the API:
flask run

Open http://127.0.0.1:5000/predict_data in your browser and enter feature values to get a prediction.

📝 Contributing
PRs welcome! Please fork the repo, create a feature branch, and submit a pull request.

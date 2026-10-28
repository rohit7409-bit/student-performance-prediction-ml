# Student Performance Prediction Using ML

Streamlit app that estimates an exam score from study hours, attendance, previous score, and assignment completion.

## Windows setup
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Upload app.py, train_model.py, requirements.txt, README.md, data/student_performance.csv, and student_model.joblib to GitHub. Deploy with Streamlit Community Cloud using `app.py` as the main file.

**Limitation:** generated dataset is synthetic and this is an academic demo, not a real student assessment tool.

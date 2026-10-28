from pathlib import Path
import random, joblib, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ROOT=Path(__file__).parent; DATA=ROOT/"data"/"student_performance.csv"; MODEL=ROOT/"student_model.joblib"
random.seed(42)
rows=[]
for _ in range(1000):
    hours=random.uniform(0,10); attendance=random.uniform(35,100); previous=random.uniform(20,100); assignments=random.uniform(0,100)
    noise=random.gauss(0,7)
    score=max(0,min(100,hours*3.2+attendance*.16+previous*.28+assignments*.12+noise-4))
    rows.append({"study_hours":round(hours,2),"attendance":round(attendance,1),"previous_score":round(previous,1),"assignments_completed":round(assignments,1),"exam_score":round(score,1)})
df=pd.DataFrame(rows); DATA.parent.mkdir(parents=True,exist_ok=True); df.to_csv(DATA,index=False)
X=df[["study_hours","attendance","previous_score","assignments_completed"]]; y=df["exam_score"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
model=RandomForestRegressor(n_estimators=120,max_depth=10,random_state=42); model.fit(Xtr,ytr); pred=model.predict(Xte)
print("Held-out synthetic test results")
print(f"MAE: {mean_absolute_error(yte,pred):.2f}")
print(f"RMSE: {mean_squared_error(yte,pred)**0.5:.2f}")
print(f"R2: {r2_score(yte,pred):.3f}")
joblib.dump(model,MODEL); print("Saved model:",MODEL); print("Note: synthetic-data metrics are for demonstration only.")

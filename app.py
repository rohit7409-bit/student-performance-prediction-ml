from pathlib import Path
import joblib, pandas as pd, streamlit as st
ROOT=Path(__file__).parent
st.set_page_config(page_title="Student Performance Predictor",page_icon="🎓")
st.title("🎓 Student Performance Predictor")
st.write("Estimate an exam score from study habits and previous performance.")
st.info("Educational demo only. Do not use predictions for real academic decisions.")
model_path=ROOT/"student_model.joblib"
if not model_path.exists():
    st.error("Model missing. Run `python train_model.py` first."); st.stop()
model=joblib.load(model_path)
c1,c2=st.columns(2)
with c1:
    hours=st.slider("Study hours per day",0.0,10.0,4.0,0.5)
    attendance=st.slider("Attendance (%)",0,100,80)
with c2:
    previous=st.slider("Previous exam score (%)",0,100,65)
    assignments=st.slider("Assignments completed (%)",0,100,75)
if st.button("Predict expected score",type="primary",use_container_width=True):
    row=pd.DataFrame([{"study_hours":hours,"attendance":attendance,"previous_score":previous,"assignments_completed":assignments}])
    score=max(0,min(100,float(model.predict(row)[0])))
    st.metric("Estimated exam score",f"{score:.1f}%")
    if score>=75: st.success("Illustrative category: High")
    elif score>=50: st.warning("Illustrative category: Moderate")
    else: st.error("Illustrative category: Needs support")
    st.caption("Trained on synthetic teaching data; the estimate is not a judgment of ability.")
with st.expander("How it works"):
    st.write("A Random Forest regression model learns from synthetic examples and estimates a score from four input features.")

import streamlit as st
import pandas as pd
import time
from datetime import datetime

ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
df = pd.read_csv("D:/Rohith/data/Attendence" + date + ".csv")

st.dataframe(df.style.highlight_min(axis=0))

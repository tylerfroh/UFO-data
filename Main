
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import pydeck as pdk
import numpy as np
import altair as alt
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics.pairwise import  cosine_similarity

from PIL import Image




st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {1100}px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )
st.markdown("""
<style>
.big-font {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)




st.title('UFO Canada Analysis')
st.header('Sneak peek into UFO sightings')
st.write("""The analysis was performed to provide in-depth analysis into the great mystery """)


df_data = pd.read_csv("C:/Repo/scrap-p/UFO.csv")


df_geo = pd.read_excel("C:/Repo/scrap-p/UFO_merge.xlsx", sheet_name="Merge1")

df_geo = pd.DataFrame(df_geo, columns=['id',  'Summary', 'Shape',
                                       'Duration','Location','lat','lon','Province'])

df_geo['Duration'] = df_geo['Duration'].astype(str)
print(df_geo)

@st.cache
def load_data(nrows):

    return df



selected_prov = st.selectbox('Pick Your Province', (df_geo['Province'].unique()))
df_selected_Data = df_geo.loc[df_geo['Province'] == selected_prov]

st.dataframe(data=df_selected_Data)
st.map(df_geo)




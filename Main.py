
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import altair as alt


from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics.pairwise import  cosine_similarity

from PIL import Image


image = Image.open('./ufo.png')
st.image(image,width=120)


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




df_geo = pd.read_excel("./UFO_merge.xlsx", sheet_name="Merge1")

df_geo = pd.DataFrame(df_geo, columns=['Summary', 'Shape',
                                       'Duration','Location','lat','lon','Province','population','dentisty'])
df_geo.drop_duplicates()

df1 = df_geo.groupby('Location').count()
print(df1)


df_geo['freq_count'] = df_geo.groupby('Location')['Location'].transform('count')


df_geo['population'] = df_geo['population'].astype(int)
df_geo['dentisty'] = df_geo['dentisty'].astype(int)
df_geo['Duration'] = df_geo['Duration'].astype(str)
df_geo["Population Density"] = df_geo["population"].div(df_geo["dentisty"].values)

df_pop_den = pd.DataFrame(df_geo, columns=['Location','Province','population','dentisty','Population Density','freq_count'])

df_pop_den.drop_duplicates()


df_pop_den["Sighting Odds %"] = df_pop_den["freq_count"].div(df_pop_den["Population Density"].values)

df_pop_den['Sighting Odds %'] = df_pop_den['Sighting Odds %'].multiply(100)


df_pop_den['Odds of Ufo Factor'] = df_pop_den['Sighting Odds %'].round(decimals = 2)


df_pop_rank_in = df_pop_den.drop_duplicates()
df_pop_rank = df_pop_rank_in.nlargest(n=25, columns=['Sighting Odds %'])
df_pop_rank['Rank'] = np.arange(len(df_pop_rank))



df_geo = df_geo.replace('\n', '', regex=True)
df_geo = df_geo.replace('-', '', regex=True)
df_geo = df_geo.replace(',', '', regex=True)
df_geo = df_geo.replace('/', '', regex=True)
df_geo = df_geo.replace('\d+', '', regex=True)
df_geo = df_geo.replace(':', ' ', regex=True)
# counting number of characters in a tweet
df_geo['characters'] = df_geo['Summary'].str.len()
# df['totalwords'] = df['COMMENTS'].str.count(' ') + 1
df_geo['totalwords'] = df_geo['Summary'].str.split().str.len()

# TWO WORD CODE
st.subheader('Frequently used words')
st.write("""The most common two and three words provide a good insight into the ufo sightings.
 The analysis conducted returns the most relevant words from the ufo sightings. """)
# TWO WORD VECTORIZER
@st.cache(allow_output_mutation=True)
def count_two_words_data(df_geo):
    cvec = CountVectorizer(stop_words = 'english', ngram_range=(2,2))
    cvec.fit(df_geo['Summary'].dropna())
    cvec_counts = cvec.transform(df_geo['Summary'].dropna())
    occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
    counts_df_two = pd.DataFrame({'term': cvec.get_feature_names(), 'occurrences': occ})
    counts_df_two.sort_values(by='occurrences', ascending=False).head(10)
    return counts_df_two
counts_df_two = count_two_words_data(df_geo)

# THREE WORD VECTORIZER
@st.cache(allow_output_mutation=True)
def count_three_words_data(df_geo):
    cvec = CountVectorizer(stop_words = 'english', ngram_range=(3,3))
    cvec.fit(df_geo['Summary'].dropna())
    cvec_counts = cvec.transform(df_geo['Summary'].dropna())
    occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
    counts_df_three = pd.DataFrame({'term': cvec.get_feature_names(), 'occurrences': occ})
    counts_df_three.sort_values(by='occurrences', ascending=False).head(20)
    return counts_df_three
counts_df_three = count_three_words_data(df_geo)

# TWO WORD CHART
alt.data_transformers.disable_max_rows()


def chart_two_words(counts_df_two):
    chart_two = (
        alt.Chart(counts_df_two)
            .mark_bar()
            .encode(
            x=alt.X('occurrences:Q', axis=alt.Axis(grid=False, labelAngle=0)),
            y=alt.Y('term', title='Two word terms',
                    sort=alt.EncodingSortField(field='occurrences', order='descending')),
            # color = alt.Color('term', legend = None, scale=alt.Scale(scheme='tealblues')),

            tooltip=['occurrences']
        )
            .properties(
            title="Most frequently occurring two words ",
            width=600,
            height=450,
        ).transform_window(
            rank='rank(occurrences)',
            sort=[alt.SortField('occurrences', order='descending')]
        ).transform_filter(
            (alt.datum.rank < 15)
        )

    ).configure_view(
        strokeOpacity=0
    ).configure_title(fontSize=18).configure_axis(
        labelFontSize=14,
        titleFontSize=16
    )
    return chart_two


col, col2, col3 = st.columns([2, 6, 1])
with col2:
    st.write(chart_two_words(counts_df_two))

# THREE WORD CHART
alt.data_transformers.disable_max_rows()


def chart_three_words(counts_df_three):
    chart_three = (
        alt.Chart(counts_df_three)
            .mark_bar()
            .encode(
            x=alt.X('occurrences:Q', axis=alt.Axis(grid=False, labelAngle=0)),
            y=alt.Y('term', title='Three word terms',
                    sort=alt.EncodingSortField(field='occurrences', order='descending')),
            # color=alt.Color('term', legend=None, scale=alt.Scale(scheme='tealblues')),

            tooltip=['occurrences']
        )
            .properties(
            title="Most frequently occurring three words ",
            width=600,
            height=450,
        ).transform_window(
            rank='rank(occurrences)',
            sort=[alt.SortField('occurrences', order='descending')]
        ).transform_filter(
            (alt.datum.rank < 13)
        )

    ).configure_view(
        strokeOpacity=0
    ).configure_title(fontSize=18).configure_axis(
        labelFontSize=14,
        titleFontSize=16
    )
    return chart_three


col1, col2, col3 = st.columns([2, 6, 2])
with col2:
    st.write(chart_three_words(counts_df_three))


@st.cache
def load_data(nrows):

    return df

#map
st.subheader('Map of sightings')
st.map(df_geo)

#population rank
st.subheader('Top 25 most likely to see a ufo based on population density')
st.dataframe(data=df_pop_rank)

st.subheader('Key word search for comments - can Filter based on province')

#TABLE SEARCH TOOL

df_search = df_geo.sort_values(by=['Location'])
field_office = df_search['Province'].unique()

with st.form('Form1'):
    col1, col2= st.columns(2)
    with col1:
        sentence = st.text_input('Input your sentence here:')
        submitted1 = st.form_submit_button('Submit')
    with col2:
        field_choice = st.selectbox(label ='Select City:',options= field_office)

    comments = df_search["Summary"].loc[df_search["Province"] == field_choice]


    if sentence:
        filtered_df = df_search.loc[(df_search['Province']==field_choice) & (df_search['Summary'].str.contains(sentence, na=False))]
        st.dataframe(filtered_df[['Summary','Location','Province']].style.set_precision(0))
        
selected_prov = st.selectbox('Pick Your Province', (df_geo['Province'].unique()))
df_selected_Data = df_geo.loc[df_geo['Province'] == selected_prov]


st.subheader('all data')
st.dataframe(data=df_selected_Data)

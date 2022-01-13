# core Pkgs
from datetime import datetime
from altair.vegalite.v4.schema.channels import Color
import streamlit as st
import altair as alt
from  plotly import express as px

# EDA Pkgs
import pandas as pd
import numpy as np

# Utils
import pickle
pipe_lr = pickle.load(open('app\model\emotion_classifier.pkl','rb'))



# Track Utils
from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table

# Emoji
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}
emotions_emoji_dict['anger']
# F
def predict_emotion(docs):
    result = pipe_lr.predict([docs])
    return result


def get_prediction_proba(docs):
    result = pipe_lr.predict_proba([docs])
    return result


# main application
def main():
    menu = ['Home',"Monitor","About"]
    choice = st.sidebar.selectbox('MENU',menu)
    create_page_visited_table()
    create_emotionclf_table()
    if choice =='Home':
        add_page_visited_details('Home',datetime.now())
        st.subheader('Home-Emotion In Text')

        with st.form(key="emotion_clf_form"):
            raw_text = st.text_area('Type Here')
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2 = st.columns(2)

            # Apply Fxn Here
            prediction1 = predict_emotion(raw_text)
            
            probability = get_prediction_proba(raw_text)

            add_prediction_details(raw_text,prediction1[0],np.max(probability),datetime.now())

            with col1:
                st.success('Original Text')
                st.write(raw_text)

                st.success('Prediction')
                emoji_icon = emotions_emoji_dict[prediction1[0]]
                st.write("{}:{}".format(prediction1[0],emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success('Prediction Probability')
                # st.write(probability)
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ['emotion',"probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotion',y= 'probability',color='emotion')
                st.altair_chart(fig,use_container_width=True)

    elif choice=="Monitor":
        add_page_visited_details('Monitor',datetime.now())
        st.subheader('Monitor App')

        with st.expander('Page Metrics'):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(),columns=['Pagename','Time of Visit'])
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Pagename',y='Counts',color='Pagename')
            st.altair_chart(c,use_container_width=True)

            p = px.pie(pg_count,values='Counts',names='Pagename')
            st.plotly_chart(p,use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):
            df_emotion = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
            st.dataframe(df_emotion)

            prediction_count = df_emotion['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x="Prediction",y='Counts',color="Prediction")
            st.altair_chart(pc,use_container_width=True)


    else:
        st.subheader('About')
        st.write('This is emotion analysis application')
        add_page_visited_details('About',datetime.now())






if __name__ == '__main__':
    main() 
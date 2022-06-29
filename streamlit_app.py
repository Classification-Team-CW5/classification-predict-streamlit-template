import streamlit as st
import joblib, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
import emoji
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import neattext.functions as nfx

def clean(tweet):
    tweet = re.sub(r'@[A-Za-z0-9_]+','',tweet) # Removing @mentions or usernames
    tweet = re.sub(r'#','',tweet) # Removing #tag symbol
    tweet = re.sub(r'RT[\s]+',' ',tweet) # Remvoing RT (Retweet)
    tweet = re.sub(r'https?:\/\/\S+','web-url',tweet) # Removing hyperlinks
    tweet = re.sub(r' +',' ',tweet) # Removing extra whitespaces

    tweet = re.sub(r':', ' ', emoji.demojize(tweet)) # Turn emojis into words

    # Removes puctuation
    punctuation = re.compile("[.;:!\'’‘“”?,\"()\[\]]")
    tweet = punctuation.sub("", tweet.lower())

    return tweet

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style/style.css")

# Vectorizer
news_vectorizer = open("resources/team_cw5_vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw_data = pd.read_csv("resources/train.csv")

# lottie urls
lottie_twitter = load_lottie_url("https://assets4.lottiefiles.com/datafiles/4pdUCLHZQdxJX79/data.json")
lottie_coding = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_vnikrcia.json")
lottie_learn = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_knvn3kk2.json")
lottie_chart = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_q5qeoo3q.json")

def main():

        selected = option_menu(
            menu_title=None,
            options=["Home","Visauls","Prediction","Kaggle","Contact"],
            icons=["house","bar-chart","code-slash","award","envelope"],
            default_index=0,
            orientation="horizontal",
        )
        if selected == "Home":
            from PIL import Image
            image = Image.open('resources/imgs/EDSA_logo.png')
            st.image(image, caption='Amazing People, Amazing Things!', use_column_width=True)

            st.title("Tweet Sentiment Analysis")
            st.write(" Machine Learning models that are able to classify whether or not a person believes in climate change, based on their novel tweet data.")

            left_column, right_column = st.columns(2)
            with left_column:
                st_lottie(lottie_twitter)
            with right_column:
                st.empty()

            st.info("General Information")
            st.markdown(""" We have deployed Machine Learning models that are able to classify 
                whether or not a person believes in climate change, based on their novel tweet data. 
                Like any data lovers, these are robust solutions to that can provide access to a 
                broad base of consumer sentiment, spanning multiple demographic and geographic categories. 
                So, do you have a Twitter API and ready to scrap? or just have some tweets off the top of your head? 
                Do explore the rest of this app's buttons.
                """)

            st.write("---")
            left_column, right_column = st.columns(2)
            with left_column:
                st_lottie(lottie_coding)
            with right_column:
                st_lottie(lottie_learn)

        if selected == "Visauls":
            left_column, right_column = st.columns(2)
            with left_column:
                st.subheader("Visaulization/Charts")
                st.info("""
                            The following are some of the charts that we have created from the raw data.
                            Some of the text is too long and may cut off, feel free to right click on the chart
                             and either save it or open it in a new window to see it properly.
                            """)
            with right_column:
                st_lottie(lottie_chart)

            st.sidebar.title("Data Analyser")
            st.sidebar.markdown("Basic analysis about our data frame")

            st.write("---")
            st.subheader("You can click on the *View raw data* button to have a look at the data frame")
            if st.checkbox("View raw data"):
                st.write(raw_data.head(50))
            st.write("---")

            st.sidebar.subheader("Sentiment Analyser")

            st.markdown("Visualisations")
            select = st.sidebar.selectbox("Visualisation Option", ["Bar graph","Pie chart"],key=1)
            values = raw_data['sentiment'].value_counts()/raw_data.shape[0]
            labels = (raw_data['sentiment'].value_counts()/raw_data.shape[0]).index
            if select == "Pie chart":
                fig = plt.figure(figsize = (10, 5))
                colors = ["orange", "blue", "purple", "pink"]
                plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.04, 0, 0, 0), colors=colors)
                st.pyplot(fig)
            
            else:
                fig = plt.figure(figsize = (10, 5))
                sns.countplot(x='sentiment' ,data = raw_data, palette='PRGn')
                plt.ylabel('Count')
                plt.xlabel('Sentiment')
                plt.title('Number of Messages Per Sentiment')
                st.pyplot(fig)

            st.sidebar.subheader("Popular Tags")
            popular = st.sidebar.selectbox("Do you want to see popular tags?",["Select option","YES"],key=1)

            st.write("---")
            if popular == "YES":
                fig = plt.figure(figsize = (10, 5))
                raw_data['users'] = [''.join(re.findall(r'@\w{,}', line)) if '@' in line else np.nan for line in raw_data.message]
                sns.countplot(y="users", hue="sentiment", data=raw_data, order=raw_data.users.value_counts().iloc[:20].index, palette='PRGn') 
                plt.ylabel('User')
                plt.xlabel('Number of Tags')
                plt.title('Top 20 Most Popular Tags')
                st.pyplot(fig)

        if selected == "Contact":
            st.subheader("Hi, this is Team CW5 :wave:")

            st.subheader("Team Supervisor")
            st.write("- Claudia Elliot Wilson")
    
            st.subheader("Team Members")
            st.write("- Thembani Maswanganyi")
            st.write("- Bongo Bokoa Seakhoa")
            st.write("- Tumishang Mankoe")
            st.write("- Kamohelo Mohlabula")
            st.write("- Patrick Parri")
            st.write("- Lerato Rafapa")

            st.subheader("Github")
            st.markdown("Click below to learn more about our organization on github")
            st.write("[Learn More >](https://github.com/Classification-Team-CW5)")


            st.subheader("Get in touch with us!")
            contact_form = """
            <form action="https://formsubmit.co/t.maswanganyi101@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here" required></textarea>
            <button type="submit">Send</button>
            </form>
            """
            left_column, right_column = st.columns(2)
            with left_column:
                st.markdown(contact_form, unsafe_allow_html=True)
            with right_column:
                st.empty()

        if selected == "Kaggle":
            st.header("Kaggle Competition")
            st.subheader("EDSA - Climate Change Belief Analysis 2022")
            st.markdown("Predict an individual’s belief in climate change based on historical tweet data")
            st.write("[Learn More >](https://www.kaggle.com/competitions/edsa-climate-change-belief-analysis-2022)")

        if selected == "Prediction":
            

            # A function to load our pickeled models
            def load_prediction_models(model_file):
                loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
                return loaded_models
            
            # Getting the predictions
            def get_keys(val,my_dict):
                for key,value in my_dict.items():
                    if val == value:
                        return key

            st.info('Make Predictions of your Tweet(s) using our ML Model')

            st.subheader('Single tweet classification')
            user_text = st.text_area("Enter Text","")
            models = ["Logistic Regression","Support Vector Classifier"]
            st.sidebar.title("Model")
            model_selection = st.sidebar.selectbox("Choose A ML Model:",models)
                
            prediction_labels = {'Anti':-1,'Neutral':0,'Pro':1,'News':2}
            if st.button("Classify"):
                tweet = clean(user_text)
                vect_text = tweet_cv.transform([tweet]).toarray()
                   
                if model_selection == "Logistic Regression":
                    predictor = load_prediction_models("resources/LogisticRegression_model.pkl")
                    prediction = predictor.predict(vect_text)
                    st.success("Tweet Submitted")

                elif model_selection == "Support Vector Classifier":
                    predictor = load_prediction_models("resources/svm.pkl")
                    prediction = predictor.predict(vect_text)
                    st.success("Tweet Submitted")

                col1, col2 = st.columns(2)
                with col1:
                    st.info("Original Tweet")
                    st.write(user_text)

                with col2:
                    st.info("Cleaned Tweet")
                    results = nfx.clean_text(tweet)
                    st.write(results)

                final_result = get_keys(prediction,prediction_labels)
                st.success("Tweet Categorized as :: {}".format(final_result))

            
                    
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
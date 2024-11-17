import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from wordcloud import WordCloud
import os

def generate_visualization_page(dataset_path):
    try:
        df = pd.read_csv(dataset_path)

        # Generate a pie chart for sentiment distribution
        sentiment_mapping = {2: 'Positive', 1: 'Neutral', 0: 'Negative'}
        df['Sentiment_Label'] = df['Sentiment'].map(sentiment_mapping)
        sentiment_counts = df['Sentiment_Label'].value_counts()

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#152335')
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, textprops={'color':'#FCDC7E'} )
        plt.savefig('./app1/static/img/sentiment_pie_chart.png', transparent=True)
        plt.close()

        # Generate a word cloud
        plt.figure(figsize=(8, 6))
        text = ' '.join(df['Description'])
        wordcloud = WordCloud(width=1600, height=1200, background_color='#152335').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig('./app1/static/img/wordcloud.png', transparent=True)
        plt.close()

    except Exception as e:
        raise RuntimeError(f"Error generating visualizations: {str(e)}")
    
def generate_visualisation(review):
    plt.figure(figsize=(8, 6))
    wordcloud = WordCloud(width=1600, height=1200, background_color='#152335').generate(review)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('./app1/static/img/wordcloud.png', transparent=True)
    plt.close()
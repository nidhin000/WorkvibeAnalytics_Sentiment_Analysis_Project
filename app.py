from taipy.gui import Gui
from cleaning import process_review
from visualization import generate_visualization_page, generate_visualisation
import pandas as pd
import torch
import torch.nn.functional as F

#from transformers import RobertaForSequenceClassification
from transformers import AutoModelForSequenceClassification


# Load the pre-trained RoBERTa model for sequence classification (3 labels: negative, neutral, positive)
#model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

# Set the model to evaluation mode
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Move the model to the selected device
model.to(device)


# Function to analyze a single review
def analyze_review(state):
    review = state.review.strip()  # Remove any leading or trailing whitespace

    if not review:  # Check if review is empty or just whitespace
        state.result1 = "Error: Review cannot be empty."
    else:
        try:
            generate_visualisation(review)
            state.wordcloud = "wordcloud.png"

            # Process the review (tokenization for RoBERTa)
            inputs = process_review(review)
            # Move inputs to the same device (GPU/CPU)
            inputs = {key: val.to(device) for key, val in inputs.items()}


            # Perform the prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1)
                print(f"Logits: {logits}")
                print(f"Probabilities: {probabilities}")

                sentiment = torch.argmax(probabilities, dim=1).item()
                print(sentiment)

            # Map prediction to sentiment
            state.result1 = f"Sentiment: {'Positive' if sentiment == 2 else 'Negative' if sentiment == 0 else 'Neutral'}"
        except Exception as e:
            state.result1 = f"Error analyzing review: {str(e)}"

# Function to handle file upload
def upload_file(state):
    if not state.file_content:  # Check if file content is uploaded
        state.result2 = "Error: No file uploaded."
    else:
        state.result2 = "File uploaded successfully. Ready for analysis."
        print(f"File uploaded successfully.")  # Confirm the upload

# Function to analyze the uploaded file
def analyze_uploaded_file(state):
    if not state.file_content:  # Check if a file path has been set
        state.result2 = "Error: No file uploaded."
        return

    # Read the file content assuming it's CSV
    try:
        df = pd.read_csv(state.file_content)

        # Rename the first column to 'Description'
        first_column = df.columns[0]
        df.rename(columns={first_column: 'Description'}, inplace=True)

        # Function to predict sentiment of a single review
        def predict_sentiment(review):
            inputs = process_review(review)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                print(logits)
                sentiment = torch.argmax(logits, dim=1).item()
                print(sentiment)
                print(model.training)
            return sentiment

        # Predict sentiment for each review
        df['Sentiment'] = df['Description'].apply(predict_sentiment)

        positive_count = len(df[df['Sentiment'] == 2])
        negative_count = len(df[df['Sentiment'] == 0])
        neutral_count = len(df[df['Sentiment'] == 1])

        # Insights
        total = len(df)
        pos_percent = (positive_count / total) * 100 if total > 0 else 0
        neg_percent = (negative_count / total) * 100 if total > 0 else 0
        neu_percent = (neutral_count / total) * 100 if total > 0 else 0

        state.result2 = f"Insights: {pos_percent:.2f}% Positive, {neg_percent:.2f}% Negative, {neu_percent:.2f}% Neutral"
        df.to_csv("reviews_visualize.csv", index=False)
        generate_visualization_page("reviews_visualize.csv")
        state.piechart = "sentiment_pie_chart.png"
        state.wordcloud = "wordcloud.png"

    except Exception as e:
        state.result2 = f"Error processing file: {str(e)}"

# Function to reset the app state
def reset_button(state):
    state.review = ""
    state.result1 = ""
    state.result2 = ""
    state.file_content = None
    state.piechart= ""
    state.wordcloud = ""

# Initialize state variables
review = ""
result1 = ""
result2 = ""
file_content = None  
img_path = "logo.png"
piechart = ""
wordcloud = ""

layout1 = """
<|part|class_name=gradient-box|>
<|text-center|
<|{img_path}|image|> 

<p>WorkPulse provides insightful analysis of employee reviews, categorizing feedback as positive, negative, or neutral.<br></br>Elevate your understanding of workplace dynamics and drive success with precise sentiment prediction.</p>
|>
"""

layout2 = """
<|layout|columns=1 6 1 6 1|

<|part|
|>

<|part|class_name=review|
<h3>Please enter the review:</h3>
<|{review}|input|textarea|rows=5|class_name=fullwidth|>
<|Analyze|button|on_action=analyze_review|class_name=buttons1|> <br></br>
<|{result1}|text|visible={result1 != ""}|class_name=result|>
|>

<|part|
<br></br>
<br></br>
<br></br>
<center><h3>Or</h3></center>
|>

<|part|class_name=review|
<h3>Upload a file with reviews:</h3>
<|{file_content}|file_selector|label=Upload dataset|extensions=.csv|on_action=upload_file|><br></br>
<|Analyze Uploaded File|button|on_action=analyze_uploaded_file|class_name=buttons2|> <br></br>
<|{result2}|text|visible={result2 != ""}|class_name=result|><br></br>
|>

<|part|
|>
<|Reset|button|on_action=reset_button|class_name=buttons3|>
|>
"""

visualization_layout = """
<|part|class_name=visual-layout
<|part|class_name=visuals|
<h2>Sentiment Distribution:</h2>
<|{piechart}|image|class_name=plots|> 
|>
<|part|class_name=visuals|
<h2>Review Word Cloud:</h2>
<|{wordcloud}|image|> 
|>
|>
"""

gui = Gui(page=layout1 + layout2 + visualization_layout, css_file='styles.css')
gui.run(use_reloader=True)

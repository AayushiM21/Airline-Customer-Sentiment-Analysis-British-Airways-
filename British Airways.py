#!/usr/bin/env python
# coding: utf-8

# In[8]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 10
page_size = 100

# List to store extracted data
reviews = []
# for i in range(1, pages + 1):
for i in range(1, pages + 1):

    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())
    
    print(f"   ---> {len(reviews)} total reviews")
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["  # Start of Unicode ranges
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002702-\U000027B0"  # Misc symbols
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)
df=pd.DataFrame()
df['reviews']=reviews
# Apply the function to the text column
df['reviews'] = df['reviews'].apply(remove_emojis)

import pandas as pd
#from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to categorize sentiment using VADER
def analyze_sentiment_vader(review):
    score = sid.polarity_scores(review)
    if score['compound'] >= 0.1:
        return 'Positive'
    elif score['compound'] <= -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Apply VADER sentiment analysis
df['sentiment'] = df['reviews'].apply(analyze_sentiment_vader)

sentiment_counts = df['sentiment'].value_counts()
import matplotlib.pyplot as plt

# Define colors and explode parameters for better visualization
colors = ['#66c2a5', '#fc8d62', '#8da0cb']  # Light green, orange, and blue colors
explode = (0.05, 0.05, 0.05)  # Slightly separate each slice

# Plot the pie chart with enhanced aesthetics
plt.figure(figsize=(5,6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, 
        colors=colors, explode=explode, pctdistance=0.85, textprops={'fontsize': 12})

# Draw a circle at the center to make it a donut chart
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
plt.gca().add_artist(centre_circle)

# Set title and add a legend with improved layout
plt.title('Sentiment Distribution of Reviews', fontsize=16, fontweight='bold', pad=20)
plt.legend(sentiment_counts.index, title="Sentiment", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()




# In[2]:


df.to_csv("C:/Users/Mahendru/OneDrive/Desktop/british airways.txt")


# In[3]:


df.isnull().sum() 
# this means that there are no null values in the reviews column


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1=pd.read_csv("C:/Users/Mahendru/Downloads/customer_booking.csv",encoding ='latin-1')
#df1.head(60)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5,6))
sns.histplot(df1['num_passengers'], bins=10, kde=True)
plt.title('Distribution of Number of Passengers')
plt.xlabel('Number of Passengers')
plt.ylabel('Frequency')
plt.show()





# In[26]:


plt.figure(figsize=(8, 5))
sns.countplot(x='sales_channel', data=df1, palette='viridis')
plt.title('Sales Channel Distribution')
plt.xlabel('Sales Channel')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[19]:


import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
trip_counts = df1['trip_type'].value_counts()

# Define colors for each category (ensure it matches the number of categories in 'trip_counts')
colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink'][:len(trip_counts)]

# Create the pie chart with enhancements
plt.figure(figsize=(5,6))
plt.pie(trip_counts, labels=trip_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)

# Set the title
plt.title('Trip Type Distribution', fontsize=14, fontweight='bold', pad=20)

# Add the legend, positioned to the right of the chart
plt.legend(trip_counts.index, title="Trip Type", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()


# In[28]:


plt.figure(figsize=(8, 5))
sns.histplot(df1['length_of_stay'], bins=20, kde=True, color='coral')
plt.title('Length of Stay Distribution')
plt.xlabel('Length of Stay (Days)')
plt.ylabel('Frequency')
plt.show()


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df1' is your DataFrame
origin_counts = df1['booking_origin'].value_counts().head(10)

# Set up the figure
plt.figure(figsize=(5,6))

# Create the bar plot with a single color for all bars
sns.barplot(x=origin_counts.index, y=origin_counts.values, color="skyblue")

# Set titles and labels
plt.title('Top 10 Booking Origins')
plt.xlabel('Booking Origin')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

# Display the plot
plt.show()


# In[32]:


plt.figure(figsize=(10, 5))
sns.barplot(x='sales_channel', y='booking_complete', data=df1, estimator=lambda x: sum(x) / len(x) * 100)
plt.title('Booking Completion Rate by Sales Channel')
plt.xlabel('Sales Channel')
plt.ylabel('Completion Rate (%)')
plt.xticks(rotation=45)
plt.show()


# In[ ]:





# In[ ]:





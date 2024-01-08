#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install --upgrade google-api-python-client


# In[2]:


from googleapiclient.discovery import build
import pandas as pd
import seaborn as sns


# In[3]:


api_key='AIzaSyDKWOkCDJBUVSTAm4hsmoaPiCNz-x4B5ZI'
channel_ids=[
    'UCt4t-jeY85JegMlZ-E5UWtA',#Aaj tak
    'UCPP3etACgdUWvizcES1dJ8Q',#News18 India
    'UCm7lHFkt2yB_WzL67aruVBQ',#Hindustan times
    'UC_gUM8rL-Lrg6O3adPW9K1g',#WION
    'UC6RJ7-PaXg6TIH2BzZfTV7w',#Times Now
    'UCIvaYmXn910QMdemBG3v1pQ',#Zee News
    'UCz8QaiQxApLq8sLNcszYyJw',#Firstpost
    'UCttspZesZIDEwwpVIgoZtWQ',#India TV
    'UChLtXXpo4Ge1ReTEboVvTDg',#Global News
    'UCtFQDgA8J8_iiwc5-KoAQlg'#ANI
            ]
youtube=build('youtube','v3',developerKey=api_key)


# # function for getting stats

# In[4]:


def Channel_details(youtube,channel_ids):
    request=youtube.channels().list(
            part='snippet,contentDetails,statistics',
            id=','.join(channel_ids))
    response=request.execute()
    return response


# In[5]:


Channel_details(youtube,channel_ids)


# In[6]:


def get_statistics(youtube,channel_ids):
    request=youtube.channels().list(
            part='snippet,contentDetails,statistics',
            id=','.join(channel_ids))
    response=request.execute()
    final_data=[]
    for i in range(len(response['items'])):
        data=dict(Channel_name=response['items'][i]['snippet']['title'],
                 Subscriber=response['items'][i]['statistics']['subscriberCount'],
                  View=response['items'][i]['statistics']['viewCount'],
                  TotalVideo=response['items'][i]['statistics']['videoCount'])
                 
        final_data.append(data)
     
    return final_data


# In[7]:


get_statistics(youtube,channel_ids)


# In[8]:


stats=get_statistics(youtube,channel_ids)


# # Showing the data using pandas

# In[9]:


import pandas as pd
tabular_form=pd.DataFrame(stats)
tabular_form


# In[10]:


tabular_form['Subscriber']=pd.to_numeric(tabular_form['Subscriber'])
tabular_form['View']=pd.to_numeric(tabular_form['View'])
tabular_form['TotalVideo']=pd.to_numeric(tabular_form['TotalVideo'])


# # Plot of Channel_name Vs Subscribers

# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size
plt.figure(figsize=(12, 6))  # Adjust the values as needed

# Create the bar plot
ax = sns.barplot(x='Channel_name', y='Subscriber', data=tabular_form)

# Rotate x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Show the plot
plt.show()


# In[12]:


sns.set(rc={'figure.figsize':(50,8)})
ans=sns.barplot(x='Channel_name',y='Subscriber',data=tabular_form)



# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the style
sns.set(style="whitegrid")

# Set the figure size
plt.figure(figsize=(16, 8))  # Adjust the width and height as needed

# Create the bar plot
ax = sns.barplot(x='Channel_name', y='Subscriber', data=tabular_form, palette="viridis")

# Rotate x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

# Add data labels on top of bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=12,
                color='black')

# Set labels and title
plt.xlabel('Channel Name', fontsize=14)
plt.ylabel('Subscribers', fontsize=14)
plt.title('Subscriber Count by Channel', fontsize=16)

# Remove spines and adjust grid
sns.despine()
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# # Plot of Channel_name Vs Total View of channel

# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the style
sns.set(style="whitegrid")

# Set the figure size
plt.figure(figsize=(16, 8))  # Adjust the width and height as needed

# Create the bar plot
ax = sns.barplot(x='Channel_name', y='View', data=tabular_form, palette="viridis")

# Rotate x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

# Add data labels on top of bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=12,
                color='black')

# Set labels and title
plt.xlabel('Channel Name', fontsize=14)
plt.ylabel('Toatal View', fontsize=14)
plt.title('Plot of Channel_name Vs Total View of channel', fontsize=16)

# Remove spines and adjust grid
sns.despine()
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[15]:


sns.set(rc={'figure.figsize':(20,8)})
ans=sns.barplot(x='Channel_name',y='View',data=tabular_form)

# Set the figure size
plt.figure(figsize=(12, 6))  # Adjust the values as needed

# Create the bar plot
ax = sns.barplot(x='Channel_name', y='View', data=tabular_form)

# Rotate x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Show the plot
plt.show()


# 
# # Channel name Vs TotalVideo

# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the style
sns.set(style="whitegrid")

# Set the figure size
plt.figure(figsize=(16, 8))  # Adjust the width and height as needed

# Create the bar plot
ax = sns.barplot(x='Channel_name', y='TotalVideo', data=tabular_form, palette="viridis")

# Rotate x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

# Add data labels on top of bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=12,
                color='black')

# Set labels and title
plt.xlabel('Channel Name', fontsize=14)
plt.ylabel('Total Video', fontsize=14)
plt.title('Channel name Vs TotalVideo by Channel', fontsize=16)

# Remove spines and adjust grid
sns.despine()
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[17]:



sns.set(rc={'figure.figsize':(20,8)})
ans=sns.barplot(x='Channel_name',y='TotalVideo',data=tabular_form)


# In[18]:


video_ids=[
    'TlI1FRnnUds',
    '0UqYzEL92SM',
    'h3afFuyE9Ts',
    'ms5IN-30Grw',
    'EcDanqLiuks',
    'qJ225Yg2pK0',
    'QlGd0w0yrdI',
    'wdPkYqG6bdI',
    'wCVtuD9yV20',
    '8j1TRkFewG4',
          ]


# # Fucntion for getting video details..

# In[19]:


def video_details(youtube,video_ids):
    
    request=youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=','.join(video_ids))
    response=request.execute()
    return response
    
    
    


# In[20]:


video_details(youtube,video_ids)


# In[21]:


def get_video_stats(youtube,video_ids):
    video_stats=[]
    request=youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=','.join(video_ids))
    response=request.execute()
    for i in range(len(response['items'])):
        data=dict(Title=response['items'][i]['snippet']['title'],
                 
                  Views=response['items'][i]['statistics']['viewCount'],
                  Likes=response['items'][i]['statistics']['likeCount'],
                  
                 Comments=response['items'][i]['statistics']['commentCount'],
                
                 )
        likes_count=data['Likes']
        views_count=data['Views']
        ratio=int(likes_count)/int(views_count)
        data['Ratio']=ratio
                 
        video_stats.append(data)
    
    
    return video_stats


# In[22]:


get_video_stats(youtube,video_ids)


# In[23]:


info=get_video_stats(youtube,video_ids)


# In[24]:


import pandas as pd
table_form=pd.DataFrame(info)
table_form


# In[25]:


table_form['Views']=pd.to_numeric(table_form['Views'])
table_form['Comments']=pd.to_numeric(table_form['Comments'])
table_form['Likes']=pd.to_numeric(table_form['Likes'])


# # Video title Vs Views on video

# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt
from googletrans import Translator

# Create a Translator object
translator = Translator()

# Function to translate titles to English
def translate_to_english(text):
    try:
        translated = translator.translate(text, src='auto', dest='en')
        return translated.text
    except:
        return text

# Abbreviate or truncate and clean video titles
table_form['Cleaned_Title'] = table_form['Title'].apply(lambda x: x[:30].replace('\n', ' ') + '...' if len(x) > 30 else x.replace('\n', ' '))
table_form['English_Title'] = table_form['Cleaned_Title'].apply(translate_to_english)

# Set the style
sns.set(style="whitegrid")

# Set the figure size
plt.figure(figsize=(16, 8))  # Adjust the width and height as needed

# Create the bar plot
ax = sns.barplot(x='English_Title', y='Views', data=table_form, palette="viridis")

# Rotate x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

# Add data labels on top of bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=12,
                color='black')

# Set labels and title
plt.xlabel('Video Title (English)', fontsize=14)
plt.ylabel('Views', fontsize=14)
plt.title('View Count by Video Title', fontsize=16)

# Remove spines and adjust grid
sns.despine()
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[27]:


sns.set(rc={'figure.figsize':(50,8)})
ans=sns.barplot(x='Title',y='Views',data=table_form)


# # Likes Vs Views

# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the style
sns.set(style="whitegrid")

# Set the figure size
plt.figure(figsize=(16, 8))  # Adjust the width and height as needed

# Create the bar plot
ax = sns.barplot(x='Likes', y='Views', data=table_form, palette="viridis")

# Rotate x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

# Add data labels on top of bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=12,
                color='black')

# Set labels and title
plt.xlabel('Likes', fontsize=14)
plt.ylabel('Views', fontsize=14)
plt.title('Likes Count Vs Views of video', fontsize=16)

# Remove spines and adjust grid
sns.despine()
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[29]:


sns.set(rc={'figure.figsize':(50,8)})
ans=sns.barplot(x='Likes',y='Views',data=table_form)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Show the plot

plt.show()


# # Video  Vs Ratio

# In[30]:


import seaborn as sns
import matplotlib.pyplot as plt
from googletrans import Translator

# Create a Translator object
translator = Translator()

# Function to translate titles to English
def translate_to_english(text):
    try:
        translated = translator.translate(text, src='auto', dest='en')
        return translated.text
    except:
        return text

# Abbreviate or truncate and clean video titles
table_form['Cleaned_Title'] = table_form['Title'].apply(lambda x: x[:30].replace('\n', ' ') + '...' if len(x) > 30 else x.replace('\n', ' '))
table_form['English_Title'] = table_form['Cleaned_Title'].apply(translate_to_english)

# Set the style
sns.set(style="whitegrid")

# Set the figure size
plt.figure(figsize=(16, 8))  # Adjust the width and height as needed

# Create the bar plot
ax = sns.barplot(x='English_Title', y='Ratio', data=table_form, palette="viridis")

# Rotate x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

# Add data labels on top of bars
for p in ax.patches:
    value = p.get_height()
    if value < 0.01:
        value_label = '< 0.01'
    else:
        value_label = format(value, '.2f')
    ax.annotate(value_label,
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=12,
                color='black')

# Set labels and title
plt.xlabel('Video Title (English)', fontsize=14)
plt.ylabel('Ratio', fontsize=14)
plt.title('Ratio of like and views with  Video Title', fontsize=16)

# Remove spines and adjust grid
sns.despine()
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[31]:


sns.set(rc={'figure.figsize':(50,8)})
ans=sns.barplot(x='Title',y='Ratio',data=table_form)


# # Video Vs Comment

# In[32]:


import seaborn as sns
import matplotlib.pyplot as plt
from googletrans import Translator

# Create a Translator object
translator = Translator()

# Function to translate titles to English
def translate_to_english(text):
    try:
        translated = translator.translate(text, src='auto', dest='en')
        return translated.text
    except:
        return text

# Abbreviate or truncate and clean video titles
table_form['Cleaned_Title'] = table_form['Title'].apply(lambda x: x[:30].replace('\n', ' ') + '...' if len(x) > 30 else x.replace('\n', ' '))
table_form['English_Title'] = table_form['Cleaned_Title'].apply(translate_to_english)

# Set the style
sns.set(style="whitegrid")

# Set the figure size
plt.figure(figsize=(16, 8))  # Adjust the width and height as needed

# Create the bar plot
ax = sns.barplot(x='English_Title', y='Comments', data=table_form, palette="viridis")

# Rotate x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

# Add data labels on top of bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=12,
                color='black')

# Set labels and title
plt.xlabel('Video Title (English)', fontsize=14)
plt.ylabel('Comments count', fontsize=14)
plt.title('comments count with  Video Title', fontsize=16)

# Remove spines and adjust grid
sns.despine()
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[33]:


sns.set(rc={'figure.figsize':(100,20)})
ans=sns.barplot(x='Title',y='Comments',data=table_form)


# # Extracting the comment from the each video

# In[34]:


pip install --upgrade pip


# In[35]:


pip install --upgrade pillow


# In[36]:


pip install --upgrade wordcloud matplotlib pillow


# In[37]:


import matplotlib
print(matplotlib.rcParams['font.sans-serif'])


# In[38]:


matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']


# In[39]:


pip install wordcloud


# In[40]:


pip install indicnlp


# In[41]:


pip install googletrans


# In[42]:


pip install --upgrade wordcloud matplotlib


# In[43]:


author_temp=[]
comment_temp=[]


# # Video 1

# In[44]:




ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='TlI1FRnnUds'
    
)
response = request.execute()
for it in response["items"]:
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    text.append(comment_text)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => "+comment_text)


# # Creating the Word Cloud of each Video

# In[45]:


from wordcloud import WordCloud
from PIL import ImageFont
import matplotlib.pyplot as plt
from PIL import Image
import re
import numpy as np

# Combine list elements into a single string
combined_text = ' '.join(text)
# pitcher=np.array(Image.open('download.jpeg'))
# Preprocess combined text
cleaned_text = re.sub(r'[^\x00-\x7F]+', '', combined_text)
cleaned_text = cleaned_text.replace('<br>', ' ')
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

# Create a WordCloud object using default fonts
wordcloud = WordCloud(
#     width=800, height=400,mask=pitcher, background_color='black', contour_width=1, contour_color='steelblue',
    width=800, height=400, background_color='white', contour_width=1, contour_color='black',
    font_path=None  # Use default fonts
).generate(cleaned_text)

# Display the WordCloud using PIL's ImageFont
# plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# # Video2

# In[46]:




ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='0UqYzEL92SM'
    
)
response = request.execute()
for it in response["items"]:
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    text.append(comment_text)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => "+comment_text)


# In[47]:


from wordcloud import WordCloud
from PIL import ImageFont
import matplotlib.pyplot as plt
from PIL import Image
import re
import numpy as np

# Combine list elements into a single string
combined_text = ' '.join(text)
# pitcher=np.array(Image.open('word2.png'))
# Preprocess combined text
cleaned_text = re.sub(r'[^\x00-\x7F]+', '', combined_text)
cleaned_text = cleaned_text.replace('<br>', ' ')
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

# Create a WordCloud object using default fonts
wordcloud = WordCloud(
#     width=800, height=400,mask=pitcher, background_color='white', contour_width=1, contour_color='steelblue',
    width=800, height=400, background_color='white', contour_width=1, contour_color='steelblue',
    font_path=None  # Use default fonts
).generate(cleaned_text)

# Display the WordCloud using PIL's ImageFont
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# # Video3
# 

# In[48]:


#wrestlers protest at jantar mantar

ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='h3afFuyE9Ts'
    
)
response = request.execute()
for it in response["items"]:
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    text.append(comment_text)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => "+comment_text)


# In[49]:


from wordcloud import WordCloud
from PIL import ImageFont
import matplotlib.pyplot as plt
from PIL import Image
import re
import numpy as np

# Combine list elements into a single string
combined_text = ' '.join(text)
# pitcher=np.array(Image.open('word2.png'))
# Preprocess combined text
cleaned_text = re.sub(r'[^\x00-\x7F]+', '', combined_text)
cleaned_text = cleaned_text.replace('<br>', ' ')
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

# Create a WordCloud object using default fonts
wordcloud = WordCloud(
#     width=800, height=400,mask=pitcher, background_color='white', contour_width=1, contour_color='steelblue',
    width=800, height=400, background_color='white', contour_width=1, contour_color='steelblue',
    font_path=None  # Use default fonts
).generate(cleaned_text)

# Display the WordCloud using PIL's ImageFont
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# # Video4

# In[50]:


#सड़क पर महिला खिलाड़ियों का ऐसा अपमान? संसद में मोदी की ताजपोशी?

ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='ms5IN-30Grw'
    
)
response = request.execute()
for it in response["items"]:
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    text.append(comment_text)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => "+comment_text)


# In[51]:


from wordcloud import WordCloud
from PIL import ImageFont
import matplotlib.pyplot as plt
from PIL import Image
import re
import numpy as np

# Combine list elements into a single string
combined_text = ' '.join(text)
# pitcher=np.array(Image.open('word2.png'))
# Preprocess combined text
cleaned_text = re.sub(r'[^\x00-\x7F]+', '', combined_text)
cleaned_text = cleaned_text.replace('<br>', ' ')
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

# Create a WordCloud object using default fonts
wordcloud = WordCloud(
#     width=800, height=400,mask=pitcher, background_color='white', contour_width=1, contour_color='steelblue',
    width=800, height=400, background_color='white', contour_width=1, contour_color='steelblue',
    font_path=None  # Use default fonts
).generate(cleaned_text)

# Display the WordCloud using PIL's ImageFont
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# # Video5

# In[52]:


#Wrestlers Protest Ruined New Parliament Inauguration: PM Modi blatantly lied on Sengol | Third Eye
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='EcDanqLiuks'
    
)
response = request.execute()
for it in response["items"]:
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    text.append(comment_text)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => "+comment_text)


# In[53]:


from wordcloud import WordCloud
from PIL import ImageFont
import matplotlib.pyplot as plt
from PIL import Image
import re
import numpy as np

# Combine list elements into a single string
combined_text = ' '.join(text)
# pitcher=np.array(Image.open('word2.png'))
# Preprocess combined text
cleaned_text = re.sub(r'[^\x00-\x7F]+', '', combined_text)
cleaned_text = cleaned_text.replace('<br>', ' ')
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

# Create a WordCloud object using default fonts
wordcloud = WordCloud(
#     width=800, height=400,mask=pitcher, background_color='white', contour_width=1, contour_color='steelblue',
    width=800, height=400, background_color='white', contour_width=1, contour_color='steelblue',
    font_path=None  # Use default fonts
).generate(cleaned_text)

# Display the WordCloud using PIL's ImageFont
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# # Video6

# In[54]:


#New Parliament & Wrestlers : राजा के संसद के बाहर पहलवानों की पिटाई !
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='qJ225Yg2pK0'
    
)
response = request.execute()
for it in response["items"]:
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    text.append(comment_text)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => "+comment_text)


# In[55]:


from wordcloud import WordCloud
from PIL import ImageFont
import matplotlib.pyplot as plt
from PIL import Image
import re
import numpy as np

# Combine list elements into a single string
combined_text = ' '.join(text)
# pitcher=np.array(Image.open('word2.png'))
# Preprocess combined text
cleaned_text = re.sub(r'[^\x00-\x7F]+', '', combined_text)
cleaned_text = cleaned_text.replace('<br>', ' ')
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

# Create a WordCloud object using default fonts
wordcloud = WordCloud(
#     width=800, height=400,mask=pitcher, background_color='white', contour_width=1, contour_color='steelblue',
    width=800, height=400, background_color='white', contour_width=1, contour_color='steelblue',
    font_path=None  # Use default fonts
).generate(cleaned_text)

# Display the WordCloud using PIL's ImageFont
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# # Video7

# In[56]:


#28 मई से शुरू राज दंड का दौर |मोदी के मेहमानों में बृजभूषण क्यों
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='QlGd0w0yrdI'
    
)
response = request.execute()
for it in response["items"]:
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    text.append(comment_text)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => "+comment_text)


# In[57]:


from wordcloud import WordCloud
from PIL import ImageFont
import matplotlib.pyplot as plt
from PIL import Image
import re
import numpy as np

# Combine list elements into a single string
combined_text = ' '.join(text)
# pitcher=np.array(Image.open('word2.png'))
# Preprocess combined text
cleaned_text = re.sub(r'[^\x00-\x7F]+', '', combined_text)
cleaned_text = cleaned_text.replace('<br>', ' ')
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

# Create a WordCloud object using default fonts
wordcloud = WordCloud(
#     width=800, height=400,mask=pitcher, background_color='white', contour_width=1, contour_color='steelblue',
    width=800, height=400, background_color='white', contour_width=1, contour_color='steelblue',
    font_path=None  # Use default fonts
).generate(cleaned_text)

# Display the WordCloud using PIL's ImageFont
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# # Video8

# In[58]:


#Satya Hindi news Bulletin सत्य हिंदी समाचार बुलेटिन । 29 मई, सुबह तक की खबरें
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='wdPkYqG6bdI'
    
)
response = request.execute()
for it in response["items"]:
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    text.append(comment_text)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => "+comment_text)


# In[59]:


from wordcloud import WordCloud
from PIL import ImageFont
import matplotlib.pyplot as plt
from PIL import Image
import re
import numpy as np

# Combine list elements into a single string
combined_text = ' '.join(text)
# pitcher=np.array(Image.open('word2.png'))
# Preprocess combined text
cleaned_text = re.sub(r'[^\x00-\x7F]+', '', combined_text)
cleaned_text = cleaned_text.replace('<br>', ' ')
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

# Create a WordCloud object using default fonts
wordcloud = WordCloud(
#     width=800, height=400,mask=pitcher, background_color='white', contour_width=1, contour_color='steelblue',
    width=800, height=400, background_color='white', contour_width=1, contour_color='steelblue',
    font_path=None  # Use default fonts
).generate(cleaned_text)

# Display the WordCloud using PIL's ImageFont
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# # Video9

# In[60]:


#संसद में सेंगोल और जंतर-मंतर पर लाठी
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='wCVtuD9yV20'
    
)
response = request.execute()
for it in response["items"]:
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    text.append(comment_text)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => "+comment_text)


# In[61]:


from wordcloud import WordCloud
from PIL import ImageFont
import matplotlib.pyplot as plt
from PIL import Image
import re
import numpy as np

# Combine list elements into a single string
combined_text = ' '.join(text)
# pitcher=np.array(Image.open('word2.png'))
# Preprocess combined text
cleaned_text = re.sub(r'[^\x00-\x7F]+', '', combined_text)
cleaned_text = cleaned_text.replace('<br>', ' ')
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

# Create a WordCloud object using default fonts
wordcloud = WordCloud(
#     width=800, height=400,mask=pitcher, background_color='white', contour_width=1, contour_color='steelblue',
    width=800, height=400, background_color='white', contour_width=1, contour_color='steelblue',
    font_path=None  # Use default fonts
).generate(cleaned_text)

# Display the WordCloud using PIL's ImageFont
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# # Video 10
# 

# In[62]:



ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='ieQFUKxkGmw'
    
)
response = request.execute()
for it in response["items"]:
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    text.append(comment_text)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => "+comment_text)


# In[63]:


from wordcloud import WordCloud
from PIL import ImageFont
import matplotlib.pyplot as plt
from PIL import Image
import re
import numpy as np

# Combine list elements into a single string
combined_text = ' '.join(text)
# pitcher=np.array(Image.open('word2.png'))
# Preprocess combined text
cleaned_text = re.sub(r'[^\x00-\x7F]+', '', combined_text)
cleaned_text = cleaned_text.replace('<br>', ' ')
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

# Create a WordCloud object using default fonts
wordcloud = WordCloud(
#     width=800, height=400,mask=pitcher, background_color='white', contour_width=1, contour_color='steelblue',
    width=800, height=400, background_color='white', contour_width=1, contour_color='steelblue',
    font_path=None  # Use default fonts
).generate(cleaned_text)

# Display the WordCloud using PIL's ImageFont
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# # Extracting the comment from the videos

# In[64]:



author_temp=[]
comment_temp=[]
ans=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='TlI1FRnnUds'
)
response = request.execute()
for it in response["items"]:
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => "+comment_text)

print()

youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='0UqYzEL92SM'
)
response = request.execute()
for it in response["items"]:
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => "+comment_text)

print()


youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='h3afFuyE9Ts'
)
response = request.execute()
for it in response["items"]:
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => "+comment_text)


print()

youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='ms5IN-30Grw'
    
)
response = request.execute()

for it in response["items"]:
#     maxResults=50
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => "+comment_text)

print()

youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='EcDanqLiuks'
    
)
response = request.execute()

for it in response["items"]:
#     maxResult=50
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => " +comment_text)
    
print()

youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='qJ225Yg2pK0'
    
)
response = request.execute()

for it in response["items"]:
#     maxResult=50
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => " +comment_text)

print()

youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='QlGd0w0yrdI'
    
)
response = request.execute()

for it in response["items"]:
#     maxResult=50
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => " +comment_text)
    
print()

youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='wdPkYqG6bdI'
    
)
response = request.execute()

for it in response["items"]:
#     maxResult=50
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => " +comment_text)
    
print()

youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='wCVtuD9yV20'
    
)
response = request.execute()

for it in response["items"]:
#     maxResult=50
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => " +comment_text)

print()

youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='ieQFUKxkGmw'
    
)
response = request.execute()

for it in response["items"]:
#     maxResult=50
    comment=it["snippet"]["topLevelComment"]
    author=comment["snippet"]["authorDisplayName"]
    comment_text=comment["snippet"]["textDisplay"]
    author_temp.append(author)
    comment_temp.append(comment_text)
    ans.append(author+" => "+comment_text)
    print(author+" => " +comment_text)

print()


# # Merging all wordclouds

# In[65]:


from wordcloud import WordCloud
from PIL import ImageFont
import matplotlib.pyplot as plt
from PIL import Image
import re
import numpy as np

# Combine list elements into a single string
combined_text = ' '.join(comment_temp)
# pitcher=np.array(Image.open('word2.png'))
# Preprocess combined text
cleaned_text = re.sub(r'[^\x00-\x7F]+', '', combined_text)
cleaned_text = cleaned_text.replace('<br>', ' ')
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

# Create a WordCloud object using default fonts
wordcloud = WordCloud(
#     width=800, height=400,mask=pitcher, background_color='white', contour_width=1, contour_color='steelblue',
    width=800, height=400, background_color='white', contour_width=1, contour_color='steelblue',
    font_path=None  # Use default fonts
).generate(cleaned_text)

# Display the WordCloud using PIL's ImageFont
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[66]:


comment_form=pd.DataFrame(ans)
comment_form


# # Sentimental Analysis of video comments

# In[67]:


pip install nltk textblob wordcloud matplotlib


# In[ ]:


import pandas as pd
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK resources (run this once)
nltk.download('vader_lexicon')
nltk.download('punkt')


# In[ ]:


import os
print(os.getcwd())


# In[ ]:


import os

# Change the working directory to the directory containing the Excel file

new_working_directory = 'C:\\Users\\saura\\OneDrive\\Desktop\\IP_APIs_Work'
os.chdir(new_working_directory)


# In[ ]:


print(os.getcwd())


# # Sheet1

# In[ ]:


import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob  
from wordcloud import WordCloud
import matplotlib.pyplot as plt


file_path = r"C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx"


# In[ ]:



sheet_name1 = 'Video1'  
df = pd.read_excel(file_path, sheet_name=sheet_name1)

df = df.dropna(subset=['Comment'])

sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    # Remove special characters, symbols, and unwanted formatting
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

def classify_sentiment_textblob(text):
    if isinstance(text, str):  
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity < 0:
            return 'negative'
        else:
            return 'neutral'
    else:
        return 'neutral'  


def classify_sentiment_vader(text):
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores['compound'] > 0.05:
        return 'positive'
    elif sentiment_scores['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'


df['Comment'] = df['Comment'].astype(str).apply(clean_text)


df['sentiment_textblob'] = df['Comment'].apply(classify_sentiment_textblob)  
df['sentiment_vader'] = df['Comment'].apply(classify_sentiment_vader)



plt.figure(figsize=(12, 6))

# Sentiment distribution using TextBlob
plt.subplot(1, 2, 1)
textblob_counts = df['sentiment_textblob'].value_counts()
textblob_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (TextBlob)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(textblob_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

# Visualize sentiment distribution using VADER
plt.subplot(1, 2, 2)
vader_counts = df['sentiment_vader'].value_counts()
vader_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (VADER)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(vader_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()
print("X-Axis: The x-axis represents the Sentiment labels, which can be 'positive', 'negative', or 'neutral'.")
print("Y-Axis: The y-axis represents the Count or frequency of comments with each sentiment label.")
print()

# Create a WordCloud from cleaned comments
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Comment']))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of YouTube Comments')
plt.show()


# # Sheet2

# In[ ]:



sheet_name2 = 'Video2'  
df = pd.read_excel(file_path, sheet_name=sheet_name2)

df = df.dropna(subset=['Comment'])


sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    # Remove special characters, symbols, and unwanted formatting
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

def classify_sentiment_textblob(text):
    if isinstance(text, str):  
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity < 0:
            return 'negative'
        else:
            return 'neutral'
    else:
        return 'neutral'  


def classify_sentiment_vader(text):
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores['compound'] > 0.05:
        return 'positive'
    elif sentiment_scores['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'


df['Comment'] = df['Comment'].astype(str).apply(clean_text)


df['sentiment_textblob'] = df['Comment'].apply(classify_sentiment_textblob)  
df['sentiment_vader'] = df['Comment'].apply(classify_sentiment_vader)

# Visualize sentiment distribution
plt.figure(figsize=(12, 6))

# Sentiment distribution using TextBlob
plt.subplot(1, 2, 1)
textblob_counts = df['sentiment_textblob'].value_counts()
textblob_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (TextBlob)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(textblob_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

# Visualize sentiment distribution using VADER
plt.subplot(1, 2, 2)
vader_counts = df['sentiment_vader'].value_counts()
vader_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (VADER)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(vader_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()
print("X-Axis: The x-axis represents the Sentiment labels, which can be 'positive', 'negative', or 'neutral'.")
print("Y-Axis: The y-axis represents the Count or frequency of comments with each sentiment label.")
print()

# Create a WordCloud from cleaned comments
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Comment']))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of YouTube Comments')
plt.show()


# # Sheet3

# In[ ]:



sheet_name3 = 'Video3'  
df = pd.read_excel(file_path, sheet_name=sheet_name3)

df = df.dropna(subset=['Comment'])


sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    # Remove special characters, symbols, and unwanted formatting
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

def classify_sentiment_textblob(text):
    if isinstance(text, str):  
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity < 0:
            return 'negative'
        else:
            return 'neutral'
    else:
        return 'neutral'  


def classify_sentiment_vader(text):
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores['compound'] > 0.05:
        return 'positive'
    elif sentiment_scores['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'


df['Comment'] = df['Comment'].astype(str).apply(clean_text)


df['sentiment_textblob'] = df['Comment'].apply(classify_sentiment_textblob)  
df['sentiment_vader'] = df['Comment'].apply(classify_sentiment_vader)

# Visualize sentiment distribution
plt.figure(figsize=(12, 6))

# Sentiment distribution using TextBlob
plt.subplot(1, 2, 1)
textblob_counts = df['sentiment_textblob'].value_counts()
textblob_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (TextBlob)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(textblob_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

# Visualize sentiment distribution using VADER
plt.subplot(1, 2, 2)
vader_counts = df['sentiment_vader'].value_counts()
vader_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (VADER)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(vader_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()
print("X-Axis: The x-axis represents the Sentiment labels, which can be 'positive', 'negative', or 'neutral'.")
print("Y-Axis: The y-axis represents the Count or frequency of comments with each sentiment label.")
print()

# Create a WordCloud from cleaned comments
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Comment']))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of YouTube Comments')
plt.show()


# # Sheet4

# In[ ]:



sheet_name4 = 'Video4'  
df = pd.read_excel(file_path, sheet_name=sheet_name4)

df = df.dropna(subset=['Comment'])


sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    # Remove special characters, symbols, and unwanted formatting
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

def classify_sentiment_textblob(text):
    if isinstance(text, str):  
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity < 0:
            return 'negative'
        else:
            return 'neutral'
    else:
        return 'neutral'  


def classify_sentiment_vader(text):
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores['compound'] > 0.05:
        return 'positive'
    elif sentiment_scores['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'


df['Comment'] = df['Comment'].astype(str).apply(clean_text)


df['sentiment_textblob'] = df['Comment'].apply(classify_sentiment_textblob)  
df['sentiment_vader'] = df['Comment'].apply(classify_sentiment_vader)

# Visualize sentiment distribution
plt.figure(figsize=(12, 6))

# Sentiment distribution using TextBlob
plt.subplot(1, 2, 1)
textblob_counts = df['sentiment_textblob'].value_counts()
textblob_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (TextBlob)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(textblob_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

# Visualize sentiment distribution using VADER
plt.subplot(1, 2, 2)
vader_counts = df['sentiment_vader'].value_counts()
vader_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (VADER)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(vader_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()
print("X-Axis: The x-axis represents the Sentiment labels, which can be 'positive', 'negative', or 'neutral'.")
print("Y-Axis: The y-axis represents the Count or frequency of comments with each sentiment label.")
print()
# Create a WordCloud from cleaned comments
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Comment']))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of YouTube Comments')
plt.show()


# # Sheet5

# In[ ]:



sheet_name5 = 'Video5'  
df = pd.read_excel(file_path, sheet_name=sheet_name5)

df = df.dropna(subset=['Comment'])


sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    # Remove special characters, symbols, and unwanted formatting
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

def classify_sentiment_textblob(text):
    if isinstance(text, str):  
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity < 0:
            return 'negative'
        else:
            return 'neutral'
    else:
        return 'neutral'  


def classify_sentiment_vader(text):
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores['compound'] > 0.05:
        return 'positive'
    elif sentiment_scores['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'


df['Comment'] = df['Comment'].astype(str).apply(clean_text)


df['sentiment_textblob'] = df['Comment'].apply(classify_sentiment_textblob)  
df['sentiment_vader'] = df['Comment'].apply(classify_sentiment_vader)

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))

# Sentiment distribution using TextBlob
plt.figure(figsize=(12, 6))

# Sentiment distribution using TextBlob
plt.subplot(1, 2, 1)
textblob_counts = df['sentiment_textblob'].value_counts()
textblob_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (TextBlob)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(textblob_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

# Visualize sentiment distribution using VADER
plt.subplot(1, 2, 2)
vader_counts = df['sentiment_vader'].value_counts()
vader_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (VADER)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(vader_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()
print("X-Axis: The x-axis represents the Sentiment labels, which can be 'positive', 'negative', or 'neutral'.")
print("Y-Axis: The y-axis represents the Count or frequency of comments with each sentiment label.")
print()

# Create a WordCloud from cleaned comments
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Comment']))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of YouTube Comments')
plt.show()


# # Sheet6

# In[ ]:



sheet_name6 = 'Video6'  
df = pd.read_excel(file_path, sheet_name=sheet_name6)

df = df.dropna(subset=['Comment'])


sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    # Remove special characters, symbols, and unwanted formatting
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

def classify_sentiment_textblob(text):
    if isinstance(text, str):  
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity < 0:
            return 'negative'
        else:
            return 'neutral'
    else:
        return 'neutral'  


def classify_sentiment_vader(text):
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores['compound'] > 0.05:
        return 'positive'
    elif sentiment_scores['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'


df['Comment'] = df['Comment'].astype(str).apply(clean_text)


df['sentiment_textblob'] = df['Comment'].apply(classify_sentiment_textblob)  
df['sentiment_vader'] = df['Comment'].apply(classify_sentiment_vader)

# Visualize sentiment distribution
plt.figure(figsize=(12, 6))

# Sentiment distribution using TextBlob
plt.subplot(1, 2, 1)
textblob_counts = df['sentiment_textblob'].value_counts()
textblob_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (TextBlob)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(textblob_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

# Visualize sentiment distribution using VADER
plt.subplot(1, 2, 2)
vader_counts = df['sentiment_vader'].value_counts()
vader_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (VADER)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(vader_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()
print("X-Axis: The x-axis represents the Sentiment labels, which can be 'positive', 'negative', or 'neutral'.")
print("Y-Axis: The y-axis represents the Count or frequency of comments with each sentiment label.")
print()
# Create a WordCloud from cleaned comments
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Comment']))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of YouTube Comments')
plt.show()


# # Sheet7

# In[ ]:



sheet_name7 = 'Video7'  
df = pd.read_excel(file_path, sheet_name=sheet_name7)

df = df.dropna(subset=['Comment'])


sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    # Remove special characters, symbols, and unwanted formatting
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

def classify_sentiment_textblob(text):
    if isinstance(text, str):  
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity < 0:
            return 'negative'
        else:
            return 'neutral'
    else:
        return 'neutral'  


def classify_sentiment_vader(text):
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores['compound'] > 0.05:
        return 'positive'
    elif sentiment_scores['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'


df['Comment'] = df['Comment'].astype(str).apply(clean_text)


df['sentiment_textblob'] = df['Comment'].apply(classify_sentiment_textblob)  
df['sentiment_vader'] = df['Comment'].apply(classify_sentiment_vader)

# Visualize sentiment distribution
plt.figure(figsize=(12, 6))

# Sentiment distribution using TextBlob
plt.subplot(1, 2, 1)
textblob_counts = df['sentiment_textblob'].value_counts()
textblob_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (TextBlob)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(textblob_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

# Visualize sentiment distribution using VADER
plt.subplot(1, 2, 2)
vader_counts = df['sentiment_vader'].value_counts()
vader_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (VADER)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(vader_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()
print("X-Axis: The x-axis represents the Sentiment labels, which can be 'positive', 'negative', or 'neutral'.")
print("Y-Axis: The y-axis represents the Count or frequency of comments with each sentiment label.")
print()

# Create a WordCloud from cleaned comments
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Comment']))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of YouTube Comments')
plt.show()


# # Sheet8

# In[ ]:



sheet_name8 = 'Video9'  
df = pd.read_excel(file_path, sheet_name=sheet_name8)

df = df.dropna(subset=['Comment'])


sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    # Remove special characters, symbols, and unwanted formatting
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

def classify_sentiment_textblob(text):
    if isinstance(text, str):  
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity < 0:
            return 'negative'
        else:
            return 'neutral'
    else:
        return 'neutral'  


def classify_sentiment_vader(text):
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores['compound'] > 0.05:
        return 'positive'
    elif sentiment_scores['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'


df['Comment'] = df['Comment'].astype(str).apply(clean_text)


df['sentiment_textblob'] = df['Comment'].apply(classify_sentiment_textblob)  
df['sentiment_vader'] = df['Comment'].apply(classify_sentiment_vader)

# Visualize sentiment distribution
plt.figure(figsize=(12, 6))

# Sentiment distribution using TextBlob
plt.subplot(1, 2, 1)
textblob_counts = df['sentiment_textblob'].value_counts()
textblob_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (TextBlob)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(textblob_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

# Visualize sentiment distribution using VADER
plt.subplot(1, 2, 2)
vader_counts = df['sentiment_vader'].value_counts()
vader_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (VADER)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(vader_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()
print("X-Axis: The x-axis represents the Sentiment labels, which can be 'positive', 'negative', or 'neutral'.")
print("Y-Axis: The y-axis represents the Count or frequency of comments with each sentiment label.")
print()

# Create a WordCloud from cleaned comments
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Comment']))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of YouTube Comments')
plt.show()


# # Sheet9

# In[ ]:



sheet_name9 = 'Video9'  
df = pd.read_excel(file_path, sheet_name=sheet_name9)

df = df.dropna(subset=['Comment'])


sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    # Remove special characters, symbols, and unwanted formatting
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

def classify_sentiment_textblob(text):
    if isinstance(text, str):  
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity < 0:
            return 'negative'
        else:
            return 'neutral'
    else:
        return 'neutral'  


def classify_sentiment_vader(text):
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores['compound'] > 0.05:
        return 'positive'
    elif sentiment_scores['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'


df['Comment'] = df['Comment'].astype(str).apply(clean_text)


df['sentiment_textblob'] = df['Comment'].apply(classify_sentiment_textblob)  
df['sentiment_vader'] = df['Comment'].apply(classify_sentiment_vader)

# Visualize sentiment distribution
plt.figure(figsize=(12, 6))

# Sentiment distribution using TextBlob
plt.subplot(1, 2, 1)
textblob_counts = df['sentiment_textblob'].value_counts()
textblob_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (TextBlob)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(textblob_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

# Visualize sentiment distribution using VADER
plt.subplot(1, 2, 2)
vader_counts = df['sentiment_vader'].value_counts()
vader_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (VADER)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for index, value in enumerate(vader_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()
print("X-Axis: The x-axis represents the Sentiment labels, which can be 'positive', 'negative', or 'neutral'.")
print("Y-Axis: The y-axis represents the Count or frequency of comments with each sentiment label.")
print()

# Create a WordCloud from cleaned comments
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Comment']))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of YouTube Comments')
plt.show()


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install --upgrade google-api-python-client


# In[5]:


from googleapiclient.discovery import build
import pandas as pd
import seaborn as sns


# In[6]:


api_key='AIzaSyDKWOkCDJBUVSTAm4hsmoaPiCNz-x4B5ZI'
channel_ids=['UCCJsQKOKArvDksacfT2ryQw',
             'UCrC8mOqJQpoB7NuIMKIS6rQ',
             'UCcBc0kafKTdcc5NAGEmA83g',
             'UCQNraktaySy_XY3EtajRnHg',
             'UCV3AkAsIDQjNQJHQgYwXjQA',
             'UC1gBk2nNZ9_VgJ5CrdQjnyQ',
             'UCMXdi-xFPcZV2tDXlHuZHgw',
             'UC2Nwwobsd0ctkzVieXZvyQA',
             'UCzLqOSZPtUKrmSEnlH4LAvw',
             'UCmB6U6fVkEaUpOI-_5uRjZA'
            ]
youtube=build('youtube','v3',developerKey=api_key)


# In[7]:


pip install nltk networkx google-api-python-client


# # function for getting stats

# In[8]:


def Channel_details(youtube,channel_ids):
    request=youtube.channels().list(
            part='snippet,contentDetails,statistics',
            id=','.join(channel_ids))
    response=request.execute()
    return response


# In[9]:


Channel_details(youtube,channel_ids)


# In[10]:


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


# In[11]:


get_statistics(youtube,channel_ids)


# In[12]:


stats=get_statistics(youtube,channel_ids)


# # Showing the data using pandas

# In[13]:


import pandas as pd
tabular_form=pd.DataFrame(stats)
tabular_form


# In[14]:


tabular_form['Subscriber']=pd.to_numeric(tabular_form['Subscriber'])
tabular_form['View']=pd.to_numeric(tabular_form['View'])
tabular_form['TotalVideo']=pd.to_numeric(tabular_form['TotalVideo'])


# # Plot of Channel_name Vs Subscribers

# In[15]:


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


# In[16]:


sns.set(rc={'figure.figsize':(50,8)})
ans=sns.barplot(x='Channel_name',y='Subscriber',data=tabular_form)



# In[17]:


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

# In[18]:


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


# In[19]:


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

# In[20]:


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


# In[21]:



sns.set(rc={'figure.figsize':(20,8)})
ans=sns.barplot(x='Channel_name',y='TotalVideo',data=tabular_form)


# In[23]:


video_ids=[
    'H9j7fCEwlNU',
    'ejG5zfznZqY',
    'qVplLJECr84',
    'RNIzSwdB8HA',
    'j_eauLKzZh4',
    'oSvTKboFlxA',
    'bPMKn2QXMRI',
    'X8ApGd7WMro',
    'bSSUN3g5QcM',
    'lbo-aBaRfzk',
          ]


# # Fucntion for getting video details..

# In[24]:


def video_details(youtube,video_ids):
    
    request=youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=','.join(video_ids))
    response=request.execute()
    return response
    
    
    


# In[25]:


video_details(youtube,video_ids)


# In[26]:


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


# In[27]:


get_video_stats(youtube,video_ids)


# In[28]:


info=get_video_stats(youtube,video_ids)


# In[29]:


import pandas as pd
table_form=pd.DataFrame(info)
table_form


# In[30]:


table_form['Views']=pd.to_numeric(table_form['Views'])
table_form['Comments']=pd.to_numeric(table_form['Comments'])
table_form['Likes']=pd.to_numeric(table_form['Likes'])


# # Video title Vs Views on video

# In[31]:


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


# In[32]:


sns.set(rc={'figure.figsize':(50,8)})
ans=sns.barplot(x='Title',y='Views',data=table_form)


# # Likes Vs Views

# In[33]:


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


# In[34]:


sns.set(rc={'figure.figsize':(50,8)})
ans=sns.barplot(x='Likes',y='Views',data=table_form)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Show the plot

plt.show()


# # Video  Vs Ratio

# In[35]:


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


# In[36]:


sns.set(rc={'figure.figsize':(50,8)})
ans=sns.barplot(x='Title',y='Ratio',data=table_form)


# # Video Vs Comment

# In[37]:


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


# In[38]:


sns.set(rc={'figure.figsize':(100,20)})
ans=sns.barplot(x='Title',y='Comments',data=table_form)


# # Extracting the comment from the each video

# In[39]:


pip install --upgrade pip


# In[40]:


pip install --upgrade pillow


# In[41]:


pip install --upgrade wordcloud matplotlib pillow


# In[42]:


import matplotlib
print(matplotlib.rcParams['font.sans-serif'])


# In[43]:


matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']


# In[44]:


pip install wordcloud


# In[45]:


pip install indicnlp


# In[46]:


pip install googletrans


# In[47]:


pip install --upgrade wordcloud matplotlib


# In[48]:


author_temp=[]
comment_temp=[]


# # Video 1

# In[62]:




ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='H9j7fCEwlNU'
    
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


# In[ ]:





# # Creating the Word Cloud of each Video

# In[63]:


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

# In[64]:




ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='ejG5zfznZqY'
    
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


# In[65]:


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

# In[66]:


#wrestlers protest at jantar mantar

ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='qVplLJECr84'
    
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


# In[67]:


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

# In[68]:


#सड़क पर महिला खिलाड़ियों का ऐसा अपमान? संसद में मोदी की ताजपोशी?

ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='RNIzSwdB8HA'
    
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


# In[69]:


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

# In[70]:


#Wrestlers Protest Ruined New Parliament Inauguration: PM Modi blatantly lied on Sengol | Third Eye
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='j_eauLKzZh4'
    
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


# In[71]:


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

# In[72]:


#New Parliament & Wrestlers : राजा के संसद के बाहर पहलवानों की पिटाई !
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='oSvTKboFlxA'
    
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


# In[73]:


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

# In[74]:


#28 मई से शुरू राज दंड का दौर |मोदी के मेहमानों में बृजभूषण क्यों
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='bPMKn2QXMRI'
    
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


# In[75]:


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

# In[76]:


#Satya Hindi news Bulletin सत्य हिंदी समाचार बुलेटिन । 29 मई, सुबह तक की खबरें
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='X8ApGd7WMro'
    
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


# In[77]:


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

# In[78]:


#संसद में सेंगोल और जंतर-मंतर पर लाठी
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='bSSUN3g5QcM'
    
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


# In[79]:


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

# In[80]:



ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='lbo-aBaRfzk'
    
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


# In[81]:


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

# In[82]:



author_temp=[]
comment_temp=[]
ans=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='H9j7fCEwlNU'
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
    videoId='ejG5zfznZqY'
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
    videoId='qVplLJECr84'
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
    videoId='RNIzSwdB8HA'
    
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
    videoId='j_eauLKzZh4'
    
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
    videoId='oSvTKboFlxA'
    
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
    videoId='bPMKn2QXMRI'
    
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
    videoId='X8ApGd7WMro'
    
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
    videoId='bSSUN3g5QcM'
    
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
    videoId='lbo-aBaRfzk'
    
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

# In[83]:


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


# In[84]:


comment_form=pd.DataFrame(ans)
comment_form


# # Sentimental Analysis of video comments

# In[85]:


pip install nltk textblob wordcloud matplotlib


# In[86]:


import pandas as pd
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK resources (run this once)
nltk.download('vader_lexicon')
nltk.download('punkt')


# In[87]:


import os
print(os.getcwd())


# In[88]:


import os

# Change the working directory to the directory containing the Excel file
new_working_directory = 'C:\\Users\\saura\\OneDrive\\Desktop\\IP_APIs_Work'
os.chdir(new_working_directory)


# In[159]:


print(os.getcwd())


# # Sheet1

# In[160]:


import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob  
from wordcloud import WordCloud
import matplotlib.pyplot as plt
file_path=r"C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Analysis.xlsx"


# In[161]:



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

# In[162]:



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

# In[163]:



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

# In[164]:



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

# In[165]:



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

# In[166]:



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

# In[167]:



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

# In[168]:



sheet_name8 = 'Video8'  
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

# In[169]:



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


# # Sheet10

# In[170]:



sheet_name9 = 'Video10'  
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


# In[84]:


pip install scipy


# In[85]:



pip install networkx


# In[86]:


pip install scipy==1.6.3


# In[87]:


pip install --upgrade networkx scipy


# In[94]:


# Selecting a fixed number of users from each sheet
fixed_num = 200

selected_users = set()
for df in video_dataframes.values():
    selected_users.update(df['Name'].dropna().head(fixed_num))

# Constructing a graph for these users
cross_graph_selected = nx.Graph()

for sheet_name, df in video_dataframes.items():
    users_in_sheet = set(df['Name']).intersection(selected_users)
    for other_sheet, other_df in video_dataframes.items():
        if sheet_name != other_sheet:
            common_users = users_in_sheet.intersection(set(other_df['Name']))
            for user in common_users:
                cross_graph_selected.add_edge(user, sheet_name, weight=len(common_users))

# Improved Visualization for Cross Analysis Graph

# Define colors for edges based on their weight
edge_colors = [d["weight"] for _, _, d in cross_graph_selected.edges(data=True)]
edge_alpha = [(5 + i) / (5 + max(edge_colors)) for i in edge_colors]  # Normalize edge opacities

plt.figure(figsize=(20, 15))

# Use a Kamada Kaway layout for clearer distinction
pos = nx.kamada_kawai_layout(cross_graph_selected)

# Draw nodes
nx.draw_networkx_nodes(cross_graph_selected, pos, node_size=300, node_color="skyblue")

# Draw edges with the defined colors and opacities
nx.draw_networkx_edges(cross_graph_selected, pos, width=0.5, edge_color=edge_colors, edge_cmap=plt.cm.Blues, alpha=edge_alpha)

# Draw labels
nx.draw_networkx_labels(cross_graph_selected, pos, font_size=9)

plt.title("Selected User Interactions for Video1 to Video10")
plt.axis("off")
plt.show()


# In[95]:


# Function to generate and display interaction graph for a specific video sheet
def generate_interaction_graph(main_sheet, video_dataframes, selected_users):
    graph = nx.Graph()
    
    main_users = set(video_dataframes[main_sheet]['Name']).intersection(selected_users)
    for sheet_name, df in video_dataframes.items():
        if main_sheet != sheet_name:
            common_users = main_users.intersection(set(df['Name']))
            for user in common_users:
                graph.add_edge(user, sheet_name, weight=len(common_users))
    
    # Visualization
    edge_colors = [d["weight"] for _, _, d in graph.edges(data=True)]
    edge_alpha = [(5 + i) / (5 + max(edge_colors)) for i in edge_colors]  # Normalize edge opacities

    plt.figure(figsize=(15, 10))
    pos = nx.kamada_kawai_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color="skyblue")
    nx.draw_networkx_edges(graph, pos, width=0.5, edge_color=edge_colors, edge_cmap=plt.cm.Blues, alpha=edge_alpha)
    nx.draw_networkx_labels(graph, pos, font_size=9)
    
    plt.title(f"User Interactions for {main_sheet}")
    plt.axis("off")
    plt.show()

# Generate interaction graphs for each sheet
for sheet_name in video_dataframes.keys():
    generate_interaction_graph(sheet_name, video_dataframes, selected_users)


# # Finding user present on more than 3 sheets

# In[100]:


from collections import Counter

# Counting the occurrence of each user across all sheets
all_users = []
for df in video_dataframes.values():
    all_users.extend(df['Name'].dropna().tolist())

user_counts = Counter(all_users)

# Filtering users present in at least 3 sheets
users_in_multiple_sheets = [user for user, count in user_counts.items() if count >= 3]

users_in_multiple_sheets


# In[98]:


# Function to generate interaction graph for users present in at least 3 sheets
def generate_selected_users_interaction_graph(main_sheet, video_dataframes, selected_users):
    graph = nx.Graph()
    
    main_users = set(video_dataframes[main_sheet]['Name']).intersection(selected_users)
    for sheet_name, df in video_dataframes.items():
        if main_sheet != sheet_name:
            common_users = main_users.intersection(set(df['Name']))
            for user in common_users:
                graph.add_edge(user, sheet_name, weight=len(common_users))
    
    # Visualization
    edge_colors = [d["weight"] for _, _, d in graph.edges(data=True)]
    edge_alpha = [(5 + i) / (5 + max(edge_colors)) for i in edge_colors]  # Normalize edge opacities

    plt.figure(figsize=(15, 10))
    pos = nx.kamada_kawai_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color="skyblue")
    nx.draw_networkx_edges(graph, pos, width=0.5, edge_color=edge_colors, edge_cmap=plt.cm.Blues, alpha=edge_alpha)
    nx.draw_networkx_labels(graph, pos, font_size=9)
    
    plt.title(f"Selected User Interactions for {main_sheet}")
    plt.axis("off")
    plt.show()

# Generate interaction graphs for each sheet for selected users
for sheet_name in video_dataframes.keys():
    generate_selected_users_interaction_graph(sheet_name, video_dataframes, users_in_multiple_sheets)


# In[97]:


from collections import Counter

# Counting the occurrence of each user across all sheets
all_users = []
for df in video_dataframes.values():
    all_users.extend(df['Name'].dropna().tolist())

user_counts = Counter(all_users)

# Filtering users present in at least 3 sheets
users_in_multiple_sheets = [user for user, count in user_counts.items() if count >= 3]

users_in_multiple_sheets

# Selecting the top 50 users who are present on the most sheets
top_users = [user for user, count in user_counts.most_common(50)]

# Function to generate interaction graph for top users
def generate_top_users_interaction_graph(main_sheet, video_dataframes, top_users):
    graph = nx.Graph()
    
    main_users = set(video_dataframes[main_sheet]['Name']).intersection(top_users)
    for sheet_name, df in video_dataframes.items():
        if main_sheet != sheet_name:
            common_users = main_users.intersection(set(df['Name']))
            for user in common_users:
                graph.add_edge(user, sheet_name, weight=len(common_users))
    
    # Visualization
    edge_colors = [d["weight"] for _, _, d in graph.edges(data=True)]
    edge_alpha = [(5 + i) / (5 + max(edge_colors)) for i in edge_colors]  # Normalize edge opacities

    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color="skyblue")
    nx.draw_networkx_edges(graph, pos, width=0.5, edge_color=edge_colors, edge_cmap=plt.cm.Blues, alpha=edge_alpha)
    nx.draw_networkx_labels(graph, pos, font_size=9)
    
    plt.title(f"Top User Interactions for {main_sheet}")
    plt.axis("off")
    plt.show()

# Generate interaction graphs for each sheet for top users
for sheet_name in video_dataframes.keys():
    generate_top_users_interaction_graph(sheet_name, video_dataframes, top_users)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





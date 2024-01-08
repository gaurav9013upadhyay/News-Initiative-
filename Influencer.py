#!/usr/bin/env python
# coding: utf-8

# In[140]:


pip install --upgrade google-api-python-client


# In[141]:


get_ipython().system('pip install --upgrade matplotlib seaborn')


# In[142]:


from googleapiclient.discovery import build
import pandas as pd
import seaborn as sns


# In[143]:


api_key='AIzaSyDKWOkCDJBUVSTAm4hsmoaPiCNz-x4B5ZI'
channel_ids=['UCs0kMbzhUYV2lhIV7xoWhoA',#The sham sharma show
             'UC5n-0ihUiOuuvZSSUnMNZLw',#Nikita thakur
             'UCbT_7qRIrw8TMH8ovjTYBJQ',#TRS clips
             'UC6WzPg6yxF9dQx2_O6R4lww',#Nitish Rajput
             'UCKop7_gs7xq5tbrDDYuplhA',#RJ ranauk
             'UCsDTy8jvHcwMvSZf_JGi-FA',#Abhi and nayu
             'UC-CSyyi47VX1lD9zyeABW3w',#Dhruv rathee
             'UCRgMIwmmh1-2k5HeTQ2cdkQ',#Gaurav thakur
             'UCQ86P4fFNN_MGgMs_Eg8PKg',#The chanayak dailogues
             'UCZjxPbi3AeB6YGKCfQ2TroQ'#The jaipur dialogues
             
            ]
youtube=build('youtube','v3',developerKey=api_key)


# In[144]:


pip install nltk networkx google-api-python-client


# # function for getting stats

# In[145]:


def Channel_details(youtube,channel_ids):
    request=youtube.channels().list(
            part='snippet,contentDetails,statistics',
            id=','.join(channel_ids))
    response=request.execute()
    return response


# In[146]:


Channel_details(youtube,channel_ids)


# In[147]:


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


# In[148]:


get_statistics(youtube,channel_ids)


# In[149]:


stats=get_statistics(youtube,channel_ids)


# # Showing the data using pandas

# In[150]:


import pandas as pd
tabular_form=pd.DataFrame(stats)
tabular_form


# In[151]:


tabular_form['Subscriber']=pd.to_numeric(tabular_form['Subscriber'])
tabular_form['View']=pd.to_numeric(tabular_form['View'])
tabular_form['TotalVideo']=pd.to_numeric(tabular_form['TotalVideo'])


# # Plot of Channel_name Vs Subscribers

# In[152]:


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


# In[153]:


sns.set(rc={'figure.figsize':(50,8)})
ans=sns.barplot(x='Channel_name',y='Subscriber',data=tabular_form)



# In[154]:


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

# In[155]:


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


# In[156]:


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

# In[157]:


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


# In[158]:



sns.set(rc={'figure.figsize':(20,8)})
ans=sns.barplot(x='Channel_name',y='TotalVideo',data=tabular_form)


# In[159]:


video_ids=[
    'TGFvOG09AjY',
    'fY2uzY5m3Rw',
    'wn8KqggvVlo',
    'FBrVlMihUZM',
    'ghD5fk25RSQ',
    'YjO3iYyCs6w',
    'i_Oi9Nlhjmc',
    's-kDCdoj88Q',
    '357zoAm-1Mk',
    'pTSj_G_FcRo',
          ]


# # Fucntion for getting video details..

# In[160]:


def video_details(youtube,video_ids):
    
    request=youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=','.join(video_ids))
    response=request.execute()
    return response
    
    
    


# In[161]:


video_details(youtube,video_ids)


# In[162]:


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


# In[163]:


get_video_stats(youtube,video_ids)


# In[164]:


info=get_video_stats(youtube,video_ids)


# In[165]:


import pandas as pd
table_form=pd.DataFrame(info)
table_form


# In[166]:


table_form['Views']=pd.to_numeric(table_form['Views'])
table_form['Comments']=pd.to_numeric(table_form['Comments'])
table_form['Likes']=pd.to_numeric(table_form['Likes'])


# # Video title Vs Views on video

# In[167]:


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


# In[168]:


sns.set(rc={'figure.figsize':(50,8)})
ans=sns.barplot(x='Title',y='Views',data=table_form)


# # Likes Vs Views

# In[169]:


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


# In[170]:


sns.set(rc={'figure.figsize':(50,8)})
ans=sns.barplot(x='Likes',y='Views',data=table_form)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Show the plot

plt.show()


# # Video  Vs Ratio

# In[171]:


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


# In[172]:


sns.set(rc={'figure.figsize':(50,8)})
ans=sns.barplot(x='Title',y='Ratio',data=table_form)


# # Video Vs Comment

# In[173]:


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


# In[174]:


sns.set(rc={'figure.figsize':(100,20)})
ans=sns.barplot(x='Title',y='Comments',data=table_form)


# # Extracting the comment from the each video

# In[175]:


pip install --upgrade pip


# In[176]:


pip install --upgrade pillow


# In[177]:


pip install --upgrade wordcloud matplotlib pillow


# In[178]:


import matplotlib
print(matplotlib.rcParams['font.sans-serif'])


# In[179]:


matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']


# In[180]:


pip install wordcloud


# In[181]:


pip install indicnlp


# In[182]:


pip install googletrans


# In[183]:


pip install --upgrade wordcloud matplotlib


# In[184]:


author_temp=[]
comment_temp=[]


# # Video 1

# In[185]:




ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='TGFvOG09AjY'
    
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

# In[186]:


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

# In[187]:




ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='fY2uzY5m3Rw'
    
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


# In[188]:


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

# In[189]:


#wrestlers protest at jantar mantar

ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='wn8KqggvVlo'
    
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


# In[190]:


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

# In[191]:


#सड़क पर महिला खिलाड़ियों का ऐसा अपमान? संसद में मोदी की ताजपोशी?

ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='FBrVlMihUZM'
    
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


# In[192]:


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

# In[193]:


#Wrestlers Protest Ruined New Parliament Inauguration: PM Modi blatantly lied on Sengol | Third Eye
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='ghD5fk25RSQ'
    
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


# In[194]:


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

# In[195]:


#New Parliament & Wrestlers : राजा के संसद के बाहर पहलवानों की पिटाई !
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='YjO3iYyCs6w'
    
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


# In[196]:


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

# In[197]:


#28 मई से शुरू राज दंड का दौर |मोदी के मेहमानों में बृजभूषण क्यों
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='i_Oi9Nlhjmc'
    
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


# In[198]:


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

# In[199]:


#Satya Hindi news Bulletin सत्य हिंदी समाचार बुलेटिन । 29 मई, सुबह तक की खबरें
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='s-kDCdoj88Q'
    
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


# In[200]:


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

# In[201]:


#संसद में सेंगोल और जंतर-मंतर पर लाठी
ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='357zoAm-1Mk'
    
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


# In[202]:


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

# In[203]:



ans=[]
text=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=50,
    videoId='pTSj_G_FcRo'
    
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


# In[204]:


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

# In[205]:



author_temp=[]
comment_temp=[]
ans=[]
youtube=build('youtube','v3',developerKey=api_key)
request = youtube.commentThreads().list(
    part="snippet,replies",
    maxResults=100,
    videoId='TGFvOG09AjY'
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
    videoId='fY2uzY5m3Rw'
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
    videoId='wn8KqggvVlo'
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
    videoId='FBrVlMihUZM'
    
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
    videoId='ghD5fk25RSQ'
    
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
    videoId='YjO3iYyCs6w'
    
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
    videoId='i_Oi9Nlhjmc'
    
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
    videoId='s-kDCdoj88Q'
    
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
    videoId='357zoAm-1Mk'
    
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
    videoId='pTSj_G_FcRo'
    
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

# In[206]:


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


# In[207]:


comment_form=pd.DataFrame(ans)
comment_form


# # Sentimental Analysis of video comments

# In[208]:


pip install nltk textblob wordcloud matplotlib


# In[209]:


import pandas as pd
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK resources (run this once)
nltk.download('vader_lexicon')
nltk.download('punkt')


# In[210]:


import os
print(os.getcwd())


# In[211]:


import os

# Change the working directory to the directory containing the Excel file
new_working_directory = 'C:\\Users\\saura\\OneDrive\\Desktop\\IP_APIs_Work'
os.chdir(new_working_directory)


# In[212]:


print(os.getcwd())


# # Sheet1

# In[213]:


import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob  
from wordcloud import WordCloud
import matplotlib.pyplot as plt
file_path=r"C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Analysis.xlsx"


# In[214]:



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

# In[215]:



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

# In[216]:



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

# In[217]:



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

# In[218]:



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

# In[219]:



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

# In[220]:



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

# In[221]:



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

# In[222]:



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

# In[223]:



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


# In[224]:


pip install scipy


# In[225]:



pip install networkx


# In[226]:


pip install scipy==1.6.3


# In[227]:


pip install --upgrade networkx scipy


# In[231]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the data from the Excel file
file_path = r"C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx"
video_dataframes = pd.read_excel(file_path, sheet_name=None)

# Filter out only the relevant sheets (Video1 to Video10)
relevant_sheets = {sheet_name: df for sheet_name, df in video_dataframes.items() if 'Video' in sheet_name}

# Initialize an empty graph for the video interactions
video_interaction_graph = nx.Graph()

# Iterate over each pair of video sheets to find and count common commenters
for sheet_name, df in relevant_sheets.items():
    commenter_names = set(df['Channel URL'].dropna())  # Set of commenters for the current sheet
    for other_sheet, other_df in relevant_sheets.items():
        if sheet_name != other_sheet:
            other_commenter_names = set(other_df['Channel URL'].dropna())  # Set of commenters for the other sheet
            # Find common commenters between the two sheets
            common_commenters = commenter_names.intersection(other_commenter_names)
            common_count = len(common_commenters)  # Number of common commenters

            # Add an edge only if there are common commenters (non-zero)
            if common_count > 0:
                video_interaction_graph.add_edge(sheet_name, other_sheet, weight=common_count)

# Visualization
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(video_interaction_graph)  # Positioning the nodes of the graph
nx.draw(video_interaction_graph, pos, with_labels=True, node_color='lightblue', node_size=3000, 
        edge_color='gray', width=2, font_size=15)

# Adding edge labels to show the number of common commenters
edge_labels = nx.get_edge_attributes(video_interaction_graph, 'weight')
nx.draw_networkx_edge_labels(video_interaction_graph, pos, edge_labels=edge_labels)

plt.title("Common Commenters of Influencer channels")
plt.axis("off")
plt.show()


# In[232]:


#News channels


# In[233]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the data from the Excel file

file_path = r"C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx"
video_dataframes = pd.read_excel(file_path, sheet_name=None)

# Filter out only the relevant sheets (Video1 to Video10)
relevant_sheets = {sheet_name: df for sheet_name, df in video_dataframes.items() if 'Video' in sheet_name}

# Initialize an empty graph for the video interactions
video_interaction_graph = nx.Graph()

# Iterate over each pair of video sheets to find and count common commenters
for sheet_name, df in relevant_sheets.items():
    commenter_names = set(df['Channel URL'].dropna())  # Set of commenters for the current sheet
    for other_sheet, other_df in relevant_sheets.items():
        if sheet_name != other_sheet:
            other_commenter_names = set(other_df['Channel URL'].dropna())  # Set of commenters for the other sheet
            # Find common commenters between the two sheets
            common_commenters = commenter_names.intersection(other_commenter_names)
            common_count = len(common_commenters)  # Number of common commenters

            # Add an edge only if there are common commenters (non-zero)
            if common_count > 0:
                video_interaction_graph.add_edge(sheet_name, other_sheet, weight=common_count)

# Visualization
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(video_interaction_graph)  # Positioning the nodes of the graph
nx.draw(video_interaction_graph, pos, with_labels=True, node_color='lightblue', node_size=3000, 
        edge_color='gray', width=2, font_size=15)

# Adding edge labels to show the number of common commenters
edge_labels = nx.get_edge_attributes(video_interaction_graph, 'weight')
nx.draw_networkx_edge_labels(video_interaction_graph, pos, edge_labels=edge_labels)

plt.title("Common Commenters of News Channels")
plt.axis("off")
plt.show()


# In[234]:


#Study Channel


# In[235]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the data from the Excel file
file_path = r"C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx"
video_dataframes = pd.read_excel(file_path, sheet_name=None)

# Filter out only the relevant sheets (Video1 to Video10)
relevant_sheets = {sheet_name: df for sheet_name, df in video_dataframes.items() if 'Video' in sheet_name}

# Initialize an empty graph for the video interactions
video_interaction_graph = nx.Graph()

# Iterate over each pair of video sheets to find and count common commenters
for sheet_name, df in relevant_sheets.items():
    commenter_names = set(df['Channel URL'].dropna())  # Set of commenters for the current sheet
    for other_sheet, other_df in relevant_sheets.items():
        if sheet_name != other_sheet:
            other_commenter_names = set(other_df['Channel URL'].dropna())  # Set of commenters for the other sheet
            # Find common commenters between the two sheets
            common_commenters = commenter_names.intersection(other_commenter_names)
            common_count = len(common_commenters)  # Number of common commenters

            # Add an edge only if there are common commenters (non-zero)
            if common_count > 0:
                video_interaction_graph.add_edge(sheet_name, other_sheet, weight=common_count)

# Visualization
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(video_interaction_graph)  # Positioning the nodes of the graph
nx.draw(video_interaction_graph, pos, with_labels=True, node_color='lightblue', node_size=3000, 
        edge_color='gray', width=2, font_size=15)

# Adding edge labels to show the number of common commenters
edge_labels = nx.get_edge_attributes(video_interaction_graph, 'weight')
nx.draw_networkx_edge_labels(video_interaction_graph, pos, edge_labels=edge_labels)

plt.title("Common Commenters of Study Channels")
plt.axis("off")
plt.show()


# In[236]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Define file paths for the three Excel files
file_paths = {
    'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
    'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
    'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
}

# Load the data from each Excel file into dictionaries of DataFrames
domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# Create network graphs for each domain
domain_graphs = {}
for domain, dfs in domain_dataframes.items():
    graph = nx.Graph()
    for sheet_name, df in dfs.items():
        if 'Video' in sheet_name:  # Filter out irrelevant sheets
            for other_sheet, other_df in dfs.items():
                if 'Video' in other_sheet and sheet_name != other_sheet:
                    # Find common commenters between sheets
                    common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
                    if len(common_commenters) > 0:
                        # Add an edge for common commenters
                        graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
    domain_graphs[domain] = graph

# Visualization
plt.figure(figsize=(20, 15))

# Define colors for each domain
color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# Draw the graphs for each domain with distinct colors
for domain, graph in domain_graphs.items():
    pos = nx.spring_layout(graph, seed=42)  # Consistent layout for all domains
    nx.draw_networkx(graph, pos, with_labels=True, node_color=color_map[domain], 
                     node_size=3000, edge_color='gray', width=2, font_size=12)

plt.title("Common Commenters Across Different Domains")
plt.axis("off")
plt.show()


# In[237]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Define file paths for the three Excel files
file_paths = {
    'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
    'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
    'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
}

# Load the data from each Excel file into dictionaries of DataFrames
domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# Create network graphs for each domain
domain_graphs = {}
for domain, dfs in domain_dataframes.items():
    graph = nx.Graph()
    for sheet_name, df in dfs.items():
        if 'Video' in sheet_name:  # Filter out irrelevant sheets
            for other_sheet, other_df in dfs.items():
                if 'Video' in other_sheet and sheet_name != other_sheet:
                    # Find common commenters between sheets
                    common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
                    if len(common_commenters) > 0:
                        # Add an edge for common commenters
                        graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
    domain_graphs[domain] = graph

# Visualization with subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 7))  # 1 row, 3 columns

# Define colors for each domain
color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# Draw the graphs for each domain in separate subplots
for (domain, graph), ax in zip(domain_graphs.items(), axs):
    pos = nx.spring_layout(graph, seed=42)  # Consistent layout for all domains
    nx.draw_networkx(graph, pos, ax=ax, with_labels=True, node_color=color_map[domain], 
                     node_size=1000, edge_color='gray', width=1, font_size=10)
    ax.set_title(f"{domain} Domain")
    ax.axis("off")

plt.suptitle("Common Commenters Across Different Domains")
plt.tight_layout()
plt.show()


# In[238]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Define file paths for the three Excel files
file_paths = {
    'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
    'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
    'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
}


# Load the data from each Excel file into dictionaries of DataFrames
domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# Create network graphs for each domain
domain_graphs = {}
for domain, dfs in domain_dataframes.items():
    graph = nx.Graph()
    for sheet_name, df in dfs.items():
        if 'Video' in sheet_name:  # Filter out irrelevant sheets
            for other_sheet, other_df in dfs.items():
                if 'Video' in other_sheet and sheet_name != other_sheet:
                    # Find common commenters between sheets
                    common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
                    if len(common_commenters) > 0:
                        # Add an edge for common commenters
                        graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
    domain_graphs[domain] = graph

# Visualization with subplots and edge weights
fig, axs = plt.subplots(1, 3, figsize=(20, 7))  # 1 row, 3 columns

# Define colors for each domain
color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# Draw the graphs for each domain in separate subplots with edge labels
for (domain, graph), ax in zip(domain_graphs.items(), axs):
    pos = nx.spring_layout(graph, seed=42)  # Consistent layout for all domains
    nx.draw_networkx(graph, pos, ax=ax, with_labels=True, node_color=color_map[domain], 
                     node_size=1000, edge_color='gray', width=1, font_size=10)
    
    # Adding edge labels for weights
    edge_weights = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, font_size=8, ax=ax)

    ax.set_title(f"{domain} Domain")
    ax.axis("off")

plt.suptitle("Common Commenters Across Different Domains")
plt.tight_layout()
plt.show()


# In[240]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Define file paths for the three Excel files
file_paths = {
    'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
    'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
    'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
}


# Load the data from each Excel file into dictionaries of DataFrames
domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# Create network graphs for each domain
domain_graphs = {}
for domain, dfs in domain_dataframes.items():
    graph = nx.Graph()
    for sheet_name, df in dfs.items():
        if 'Video' in sheet_name and 'Channel URL' in df.columns:  # Filter out irrelevant sheets and check 'Name' column
            for other_sheet, other_df in dfs.items():
                if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
                    # Find common commenters between sheets
                    common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
                    if len(common_commenters) > 0:
                        # Add an edge for common commenters
                        graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
    domain_graphs[domain] = graph

# Collecting all commenters from each domain
all_commenters = {}
for domain, dfs in domain_dataframes.items():
    domain_commenters = set()
    for df in dfs.values():
        if 'Channel URL' in df.columns:  # Check if 'Name' column is present
            domain_commenters.update(df['Channel URL'].dropna())
    all_commenters[domain] = domain_commenters

# Create a separate graph for cross-domain connections
cross_domain_graph = nx.Graph()

# Define colors for each domain and cross-domain edges
color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}
cross_domain_edge_color = 'red'

# Add cross-domain edges
for domain1, commenters1 in all_commenters.items():
    for domain2, commenters2 in all_commenters.items():
        if domain1 != domain2:
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                cross_domain_graph.add_edge(domain1, domain2, weight=len(common_commenters))

# Visualization with subplots for intra-domain and a separate subplot for cross-domain connections
fig, axs = plt.subplots(1, 4, figsize=(25, 7))  # 1 row, 4 columns

# Draw the intra-domain graphs in the first three subplots
for (domain, graph), ax in zip(domain_graphs.items(), axs[:3]):
    pos = nx.spring_layout(graph, seed=42)  # Consistent layout for all domains
    nx.draw_networkx(graph, pos, ax=ax, with_labels=True, node_color=color_map[domain], 
                     node_size=1000, edge_color='gray', width=1, font_size=10)
    
    # Adding edge labels for weights within domains
    edge_weights = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, font_size=8, ax=ax)

    ax.set_title(f"{domain} Domain")
    ax.axis("off")

# Draw the cross-domain graph in the fourth subplot
ax_cross_domain = axs[3]
pos_cross_domain = nx.spring_layout(cross_domain_graph, seed=42)
nx.draw_networkx(cross_domain_graph, pos_cross_domain, ax=ax_cross_domain, with_labels=True, 
                 node_size=1000, edge_color=cross_domain_edge_color, width=1, font_size=10)

# Adding edge labels for weights between domains
edge_weights_cross_domain = nx.get_edge_attributes(cross_domain_graph, 'weight')
nx.draw_networkx_edge_labels(cross_domain_graph, pos_cross_domain, edge_labels=edge_weights_cross_domain, font_size=8, ax=ax_cross_domain)

ax_cross_domain.set_title("Cross-Domain Connections")
ax_cross_domain.axis("off")

plt.suptitle("Common Commenters Within and Between Different Domains")
plt.tight_layout()
plt.show()


# In[241]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Define file paths for the three Excel files
file_paths = {
    'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
    'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
    'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
}

# Load the data from each Excel file into dictionaries of DataFrames
domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# Create network graphs for each domain using 'Channel URL'
domain_graphs_url = {}
for domain, dfs in domain_dataframes.items():
    graph = nx.Graph()
    for sheet_name, df in dfs.items():
        if 'Video' in sheet_name and 'Channel URL' in df.columns:  # Check for the right column
            for other_sheet, other_df in dfs.items():
                if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
                    # Find common commenters using 'Channel URL'
                    common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
                    if len(common_commenters) > 0:
                        graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
    domain_graphs_url[domain] = graph

# Collecting all commenters from each domain using 'Channel URL'
all_commenters_url = {}
for domain, dfs in domain_dataframes.items():
    domain_commenters = set()
    for df in dfs.values():
        if 'Channel URL' in df.columns:  # Check if 'Channel URL' column is present
            domain_commenters.update(df['Channel URL'].dropna())
    all_commenters_url[domain] = domain_commenters

# Create a separate graph for cross-domain connections using 'Channel URL'
cross_domain_graph_url = nx.Graph()
cross_domain_edge_color = 'red'

# Add cross-domain edges
for domain1, commenters1 in all_commenters_url.items():
    for domain2, commenters2 in all_commenters_url.items():
        if domain1 != domain2:
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                cross_domain_graph_url.add_edge(domain1, domain2, weight=len(common_commenters))

# Visualization with subplots for intra-domain and a separate subplot for cross-domain connections
fig, axs = plt.subplots(1, 4, figsize=(25, 7))  # 1 row, 4 columns

# Define colors for each domain
color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# Draw the intra-domain graphs in the first three subplots
for (domain, graph), ax in zip(domain_graphs_url.items(), axs[:3]):
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx(graph, pos, ax=ax, with_labels=True, node_color=color_map[domain], 
                     node_size=1000, edge_color='gray', width=1, font_size=10)
    
    # Adding edge labels for weights within domains
    edge_weights = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, font_size=8, ax=ax)

    ax.set_title(f"{domain} Domain")
    ax.axis("off")

# Draw the cross-domain graph in the fourth subplot
ax_cross_domain = axs[3]
pos_cross_domain = nx.spring_layout(cross_domain_graph_url, seed=42)
nx.draw_networkx(cross_domain_graph_url, pos_cross_domain, ax=ax_cross_domain, with_labels=True, 
                 node_size=1000, edge_color=cross_domain_edge_color, width=1, font_size=10)

# Adding edge labels for weights between domains
edge_weights_cross_domain = nx.get_edge_attributes(cross_domain_graph_url, 'weight')
nx.draw_networkx_edge_labels(cross_domain_graph_url, pos_cross_domain, edge_labels=edge_weights_cross_domain, font_size=8, ax=ax_cross_domain)

ax_cross_domain.set_title("Cross-Domain Connections")
ax_cross_domain.axis("off")

plt.suptitle("Common Commenters (Based on Channel URL) Within and Between Different Domains")
plt.tight_layout()
plt.show()


# In[242]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Define file paths for the three Excel files
file_paths = {
    'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
    'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
    'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
}

# Load the data from each Excel file into dictionaries of DataFrames
domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# Create network graphs for each domain using 'Channel URL'
domain_graphs_url = {}
for domain, dfs in domain_dataframes.items():
    graph = nx.Graph()
    for sheet_name, df in dfs.items():
        if 'Video' in sheet_name and 'Channel URL' in df.columns:  # Check for the right column
            for other_sheet, other_df in dfs.items():
                if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
                    # Find common commenters using 'Channel URL'
                    common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
                    if len(common_commenters) > 0:
                        graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
    domain_graphs_url[domain] = graph

# Collecting all commenters for each video across all domains using 'Channel URL'
video_commenters = {}
for domain, dfs in domain_dataframes.items():
    for video, df in dfs.items():
        if 'Video' in video and 'Channel URL' in df.columns:
            video_commenters[(video, domain)] = set(df['Channel URL'].dropna())

# Create a combined graph for all videos from all domains
combined_video_graph = nx.Graph()

# Define colors for each domain and inter-domain edges
color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}
inter_domain_edge_color = 'purple'

# Add intra-domain edges (within the same domain)
for domain, graph in domain_graphs_url.items():
    for u, v, data in graph.edges(data=True):
        combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'], color=color_map[domain])

# Add inter-domain edges (between videos of different domains)
for video1, commenters1 in video_commenters.items():
    for video2, commenters2 in video_commenters.items():
        if video1[1] != video2[1]:  # Ensure videos are from different domains
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                combined_video_graph.add_edge(video1, video2, weight=len(common_commenters), color=inter_domain_edge_color)

# Visualization with a single graph
plt.figure(figsize=(20, 15))
pos = nx.spring_layout(combined_video_graph, seed=42)  # Layout for visual clarity
edges = combined_video_graph.edges(data=True)

# Draw nodes with correct domain-based coloring
nx.draw_networkx_nodes(combined_video_graph, pos, node_size=1000, 
                       node_color=[color_map[node[1]] for node in combined_video_graph.nodes()])

# Draw intra-domain edges
nx.draw_networkx_edges(combined_video_graph, pos, 
                       edgelist=[(u, v) for u, v, d in edges if 'color' in d and d['color'] != inter_domain_edge_color], 
                       width=1)

# Draw inter-domain edges
nx.draw_networkx_edges(combined_video_graph, pos, 
                       edgelist=[(u, v) for u, v, d in edges if 'color' in d and d['color'] == inter_domain_edge_color], 
                       edge_color=inter_domain_edge_color, style='dashed', width=1)

# Edge labels
nx.draw_networkx_edge_labels(combined_video_graph, pos, 
                             edge_labels={(u, v): d['weight'] for u, v, d in edges}, font_size=8)

# Node labels
nx.draw_networkx_labels(combined_video_graph, pos, 
                        labels={n: n[0] for n in combined_video_graph.nodes()}, font_size=10)

plt.title("Common Commenters (Based on Channel URL) Within and Between Videos of Different Domains")
plt.axis("off")
plt.show()


# In[243]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Define file paths for the three Excel files
file_paths = {
    'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
    'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
    'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
}

# Load the data from each Excel file into dictionaries of DataFrames
domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# Create network graphs for each domain using 'Channel URL'
domain_graphs_url = {}
for domain, dfs in domain_dataframes.items():
    graph = nx.Graph()
    for sheet_name, df in dfs.items():
        if 'Video' in sheet_name and 'Channel URL' in df.columns:
            for other_sheet, other_df in dfs.items():
                if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
                    common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
                    if len(common_commenters) > 0:
                        graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
    domain_graphs_url[domain] = graph

# Collecting all commenters for each video across all domains using 'Channel URL'
video_commenters = {}
for domain, dfs in domain_dataframes.items():
    for video, df in dfs.items():
        if 'Video' in video and 'Channel URL' in df.columns:
            video_commenters[(video, domain)] = set(df['Channel URL'].dropna())

# Create a combined graph for all videos from all domains
combined_video_graph = nx.Graph()

# Define colors for each domain and inter-domain edges
color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}
inter_domain_edge_color = 'purple'

# Add intra-domain edges (within the same domain)
for domain, graph in domain_graphs_url.items():
    for u, v, data in graph.edges(data=True):
        combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'], color=color_map[domain])

# Add inter-domain edges (between videos of different domains)
for video1, commenters1 in video_commenters.items():
    for video2, commenters2 in video_commenters.items():
        if video1[1] != video2[1]:
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                combined_video_graph.add_edge(video1, video2, weight=len(common_commenters), color=inter_domain_edge_color)

# Function to calculate positions for each domain in a triangular layout
def calculate_domain_positions(graph, domain, offset, layout_func=nx.spring_layout):
    domain_nodes = [node for node in graph.nodes if node[1] == domain]
    subgraph = graph.subgraph(domain_nodes)
    pos = layout_func(subgraph, seed=42)
    pos = {node: (x + offset[0], y + offset[1]) for node, (x, y) in pos.items()}
    return pos

# Prepare a combined position dictionary for all nodes
combined_pos = {}

# Calculate positions for each domain with an offset to arrange them in a triangular layout
offsets = {'Influencer': (0, 0), 'News': (1, 1), 'Study': (2, 0)}
for domain in domain_graphs_url.keys():
    domain_pos = calculate_domain_positions(combined_video_graph, domain, offsets[domain])
    combined_pos.update(domain_pos)

# Visualization with triangular layout
plt.figure(figsize=(20, 15))
edges = combined_video_graph.edges(data=True)

# Draw nodes with domain-based coloring
nx.draw_networkx_nodes(combined_video_graph, combined_pos, node_size=1000, 
                       node_color=[color_map[node[1]] for node in combined_video_graph.nodes()])

# Draw intra-domain edges
nx.draw_networkx_edges(combined_video_graph, combined_pos, 
                       edgelist=[(u, v) for u, v, d in edges if 'color' in d and d['color'] != inter_domain_edge_color], 
                       width=1)

# Draw inter-domain edges
nx.draw_networkx_edges(combined_video_graph, combined_pos, 
                       edgelist=[(u, v) for u, v, d in edges if 'color' in d and d['color'] == inter_domain_edge_color], 
                       edge_color=inter_domain_edge_color, style='dashed', width=1)

# Edge labels
nx.draw_networkx_edge_labels(combined_video_graph, combined_pos, 
                             edge_labels={(u, v): d['weight'] for u, v, d in edges}, font_size=8)

# Node labels
nx.draw_networkx_labels(combined_video_graph, combined_pos, 
                        labels={n: n[0] for n in combined_video_graph.nodes()}, font_size=10)

plt.title("Common Commenters (Based on Channel URL) Within and Between Videos of Different Domains")
plt.axis("off")
plt.show()


# In[244]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the data from each Excel file into dictionaries of DataFrames
file_paths = {
    'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
    'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
    'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
}
domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# Create network graphs for each domain using 'Channel URL'
domain_graphs_url = {}
for domain, dfs in domain_dataframes.items():
    graph = nx.Graph()
    for sheet_name, df in dfs.items():
        if 'Video' in sheet_name and 'Channel URL' in df.columns:
            for other_sheet, other_df in dfs.items():
                if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
                    common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
                    if len(common_commenters) > 0:
                        graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
    domain_graphs_url[domain] = graph

# Collecting all commenters for each video across all domains using 'Channel URL'
video_commenters = {}
for domain, dfs in domain_dataframes.items():
    for video, df in dfs.items():
        if 'Video' in video and 'Channel URL' in df.columns:
            video_commenters[(video, domain)] = set(df['Channel URL'].dropna())

# Define specific colors for edges between each pair of domains
inter_domain_edge_colors = {
    ('Influencer', 'Study'): 'purple',
    ('Influencer', 'News'): 'orange',
    ('Study', 'News'): 'green'
}

# Define colors for each domain
color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# Create a combined graph for all videos from all domains
combined_video_graph = nx.Graph()

# Add intra-domain edges (within the same domain)
for domain, graph in domain_graphs_url.items():
    for u, v, data in graph.edges(data=True):
        combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'], color=color_map[domain])

# Add inter-domain edges (between videos of different domains)
for video1, commenters1 in video_commenters.items():
    for video2, commenters2 in video_commenters.items():
        if video1[1] != video2[1]:  # Only for different domains
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                # Determine the color based on the pair of domains
                domain_pair = tuple(sorted([video1[1], video2[1]]))
                edge_color = inter_domain_edge_colors.get(domain_pair, 'black')  # Default to black if pair not defined
                combined_video_graph.add_edge(video1, video2, weight=len(common_commenters), color=edge_color)

# Visualization with triangular layout and distinct inter-domain edge colors
plt.figure(figsize=(20, 15))
edges = combined_video_graph.edges(data=True)

# Draw nodes with domain-based coloring
nx.draw_networkx_nodes(combined_video_graph, pos, node_size=1000, 
                       node_color=[color_map[node[1]] for node in combined_video_graph.nodes()])

# Draw intra-domain edges
nx.draw_networkx_edges(combined_video_graph, pos, 
                       edgelist=[(u, v) for u, v, d in edges if 'color' in d and d['color'] in color_map.values()], 
                       width=1)

# Draw inter-domain edges with distinct colors
for domain_pair, edge_color in inter_domain_edge_colors.items():
    nx.draw_networkx_edges(combined_video_graph, pos, 
                           edgelist=[(u, v) for u, v, d in edges if 'color' in d and d['color'] == edge_color], 
                           edge_color=edge_color, style='dashed', width=1)

# Edge labels
nx.draw_networkx_edge_labels(combined_video_graph, pos, 
                             edge_labels={(u, v): d['weight'] for u, v, d in edges}, font_size=8)

# Node labels
nx.draw_networkx_labels(combined_video_graph, pos, 
                        labels={n: n[0] for n in combined_video_graph.nodes()}, font_size=10)

plt.title("Common Commenters (Based on Channel URL) Within and Between Videos of Different Domains")
plt.axis("off")
plt.show()


# In[260]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Define file paths for the three Excel files
file_paths = {
    'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
    'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
    'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
}

# Load the data from each Excel file into dictionaries of DataFrames
domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# Create network graphs for each domain using 'Channel URL'
domain_graphs_url = {}
for domain, dfs in domain_dataframes.items():
    graph = nx.Graph()
    for sheet_name, df in dfs.items():
        if 'Video' in sheet_name and 'Channel URL' in df.columns:
            for other_sheet, other_df in dfs.items():
                if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
                    common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
                    if len(common_commenters) > 0:
                        graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
    domain_graphs_url[domain] = graph

# Collecting all commenters for each video across all domains using 'Channel URL'
video_commenters = {}
for domain, dfs in domain_dataframes.items():
    for video, df in dfs.items():
        if 'Video' in video and 'Channel URL' in df.columns:
            video_commenters[(video, domain)] = set(df['Channel URL'].dropna())

# Define specific colors for edges between each pair of domains
inter_domain_edge_colors = {
    ('Influencer', 'Study'): 'purple',
    ('Influencer', 'News'): 'orange',
    ('Study', 'News'): 'black'  # Ensuring Study-News edges are orange
}

# Define colors for each domain
color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# Create a combined graph for all videos from all domains
combined_video_graph = nx.Graph()

# Add intra-domain edges (within the same domain)
for domain, graph in domain_graphs_url.items():
    for u, v, data in graph.edges(data=True):
        combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'], color=color_map[domain])

# Add inter-domain edges (between videos of different domains)
for video1, commenters1 in video_commenters.items():
    for video2, commenters2 in video_commenters.items():
        if video1[1] != video2[1]:  # Only for different domains
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                # Sort the domain pair to match the keys in the inter_domain_edge_colors
                domain_pair = tuple(sorted([video1[1], video2[1]]))
                # Use the get method to provide a default color ('black') if the key doesn't exist
                edge_color = inter_domain_edge_colors.get(domain_pair, 'black')
                combined_video_graph.add_edge(video1, video2, weight=len(common_commenters), color=edge_color)

# Function to arrange nodes of each domain in a triangular layout
def arrange_in_triangle(graph, domain_positions, domain):
    nodes = [node for node in graph.nodes if node[1] == domain]
    num_nodes = len(nodes)
    angle_step = np.pi / 2 / (num_nodes - 1)  # Spread nodes evenly
    positions = {}
    start_angle = np.pi / 4
    radius = 1.5  # Increase radius for better spread
    for i, node in enumerate(sorted(nodes)):  # Sort nodes for consistent ordering
        angle = start_angle + i * angle_step
        positions[node] = (np.cos(angle) * radius + domain_positions[domain][0], 
                           np.sin(angle) * radius + domain_positions[domain][1])
    return positions

# Vertices of an equilateral triangle for the three domains
domain_vertices = {
    'Influencer':(0,0),
    'News':(1,np.sqrt(3)/2),
    'Study': (2, 0)
}

# Calculate positions for each domain's nodes
all_positions = {}
for domain in domain_vertices:
    domain_pos = arrange_in_triangle(combined_video_graph, domain_vertices, domain)
    all_positions.update(domain_pos)

# Visualization with triangular layout and distinct inter-domain edge colors
plt.figure(figsize=(20, 15))
edges = combined_video_graph.edges(data=True)

# Draw nodes with domain-based coloring
nx.draw_networkx_nodes(combined_video_graph, all_positions, node_size=1000, 
                       node_color=[color_map[node[1]] for node in combined_video_graph.nodes()])

# Draw intra-domain edges
nx.draw_networkx_edges(combined_video_graph, all_positions, 
                       edgelist=[(u, v) for u, v, d in edges if d['color'] in color_map.values()], 
                       width=1)

# Draw all inter-domain edges with distinct colors for each pair of domains
for domain_pair, edge_color in inter_domain_edge_colors.items():
    nx.draw_networkx_edges(combined_video_graph, all_positions, 
                           edgelist=[(u, v) for u, v, d in edges if d['color'] == edge_color], 
                           edge_color=edge_color, style='dashed', width=2)

# Edge labels
nx.draw_networkx_edge_labels(combined_video_graph, all_positions, 
                             edge_labels={(u, v): d['weight'] for u, v, d in edges}, font_size=8)

# Node labels
nx.draw_networkx_labels(combined_video_graph, all_positions, 
                        labels={n: n[0] for n in combined_video_graph.nodes()}, font_size=10)

plt.title("Common Commenters (Based on Channel URL) Within and Between Videos of Different Domains")
plt.axis("off")
plt.show()


# In[261]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Define file paths for the three Excel files
file_paths = {
    'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
    'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
    'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
}

# Load the data from each Excel file into dictionaries of DataFrames
domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# Create network graphs for each domain using 'Channel URL'
domain_graphs_url = {}
for domain, dfs in domain_dataframes.items():
    graph = nx.Graph()
    for sheet_name, df in dfs.items():
        if 'Video' in sheet_name and 'Channel URL' in df.columns:
            for other_sheet, other_df in dfs.items():
                if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
                    common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
                    if len(common_commenters) > 0:
                        graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
    domain_graphs_url[domain] = graph

# Collecting all commenters for each video across all domains using 'Channel URL'
video_commenters = {}
for domain, dfs in domain_dataframes.items():
    for video, df in dfs.items():
        if 'Video' in video and 'Channel URL' in df.columns:
            video_commenters[(video, domain)] = set(df['Channel URL'].dropna())

# Define specific colors for edges between each pair of domains
inter_domain_edge_colors = {
    ('Influencer', 'Study'): 'purple',
    ('Influencer', 'News'): 'orange',
    ('Study', 'News'): 'black'
}

# Define colors for each domain
color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# Create a combined graph for all videos from all domains
combined_video_graph = nx.Graph()

# Add intra-domain edges
for domain, graph in domain_graphs_url.items():
    for u, v, data in graph.edges(data=True):
        combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'], color=color_map[domain])

# Add inter-domain edges
for video1, commenters1 in video_commenters.items():
    for video2, commenters2 in video_commenters.items():
        if video1[1] != video2[1]:  # Only for different domains
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                domain_pair = tuple(sorted([video1[1], video2[1]]))
                edge_color = inter_domain_edge_colors.get(domain_pair, 'black')
                combined_video_graph.add_edge(video1, video2, weight=len(common_commenters), color=edge_color)

# Arrange nodes in a circular layout within big circles
def arrange_in_circle(graph, domain_positions, domain, radius=1):
    nodes = [node for node in graph.nodes if node[1] == domain]
    num_nodes = len(nodes)
    angle_step = 2 * np.pi / num_nodes
    positions = {}
    for i, node in enumerate(nodes):
        angle = i * angle_step
        positions[node] = (np.cos(angle) * radius + domain_positions[domain][0], 
                           np.sin(angle) * radius + domain_positions[domain][1])
    return positions

# Big circle centers for the three domains
domain_centers = {
    'Influencer': (0, 0),
    'News': (4, 0),
    'Study': (8, 0)
}

# Calculate positions for each domain's nodes
all_positions = {}
for domain in domain_centers:
    domain_pos = arrange_in_circle(combined_video_graph, domain_centers, domain)
    all_positions.update(domain_pos)

# Visualization
plt.figure(figsize=(20, 15))
edges = combined_video_graph.edges(data=True)

# Draw big circles for each domain
for domain, center in domain_centers.items():
    circle = plt.Circle(center, 1, color=color_map[domain], fill=False)
    plt.gca().add_patch(circle)

# Draw nodes with domain-based coloring
nx.draw_networkx_nodes(combined_video_graph, all_positions, node_size=1000, 
                       node_color=[color_map[node[1]] for node in combined_video_graph.nodes()])

# Draw intra-domain edges
nx.draw_networkx_edges(combined_video_graph, all_positions, 
                       edgelist=[(u, v) for u, v, d in edges if d['color'] in color_map.values()], 
                       width=1)

# Draw inter-domain edges with distinct colors
for domain_pair, edge_color in inter_domain_edge_colors.items():
    nx.draw_networkx_edges(combined_video_graph, all_positions, 
                           edgelist=[(u, v) for u, v, d in edges if d['color'] == edge_color], 
                           edge_color=edge_color, style='dashed', width=2)

# Edge labels
nx.draw_networkx_edge_labels(combined_video_graph, all_positions, 
                             edge_labels={(u, v): d['weight'] for u, v, d in edges}, font_size=8)

# Node labels
nx.draw_networkx_labels(combined_video_graph, all_positions, 
                        labels={n: n[0] for n in combined_video_graph.nodes()}, font_size=10)

plt.title("Common Commenters Within and Between Videos of Different Domains")
plt.axis("off")
plt.show()


# In[262]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# [Your existing data loading code]

# Collecting all commenters from each domain using 'Channel URL'
all_commenters_url = {}
for domain, dfs in domain_dataframes.items():
    domain_commenters = set()
    for df in dfs.values():
        if 'Channel URL' in df.columns:  # Check if 'Channel URL' column is present
            domain_commenters.update(df['Channel URL'].dropna())
    all_commenters_url[domain] = domain_commenters

# Create a separate graph for cross-domain connections using 'Channel URL'
cross_domain_graph_url = nx.Graph()
cross_domain_edge_color = 'red'

# Add cross-domain edges
for domain1, commenters1 in all_commenters_url.items():
    for domain2, commenters2 in all_commenters_url.items():
        if domain1 != domain2:
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                cross_domain_graph_url.add_edge(domain1, domain2, weight=len(common_commenters))

# Visualization for cross-domain connections
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

pos_cross_domain = nx.spring_layout(cross_domain_graph_url, seed=42)
nx.draw_networkx(cross_domain_graph_url, pos_cross_domain, with_labels=True, 
                 node_size=1000, edge_color=cross_domain_edge_color, width=1, font_size=10)

# Adding edge labels for weights between domains
edge_weights_cross_domain = nx.get_edge_attributes(cross_domain_graph_url, 'weight')
nx.draw_networkx_edge_labels(cross_domain_graph_url, pos_cross_domain, edge_labels=edge_weights_cross_domain, font_size=8)

plt.title("Cross-Domain Connections Based on Common Commenters")
plt.axis("off")
plt.show()


# In[263]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Define file paths for the three Excel files
file_paths = {
    'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
    'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
    'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
}

# Load the data from each Excel file into dictionaries of DataFrames
domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# Create network graphs for each domain using 'Channel URL'
domain_graphs_url = {}
for domain, dfs in domain_dataframes.items():
    graph = nx.Graph()
    for sheet_name, df in dfs.items():
        if 'Video' in sheet_name and 'Channel URL' in df.columns:
            for other_sheet, other_df in dfs.items():
                if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
                    common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
                    if len(common_commenters) > 0:
                        graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
    domain_graphs_url[domain] = graph

# Define specific colors for edges between each pair of domains
inter_domain_edge_colors = {
    ('Influencer', 'Study'): 'purple',
    ('Influencer', 'News'): 'orange',
    ('Study', 'News'): 'black'
}

# Define colors for each domain
color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# [Rest of your data processing code]

# Manually add inter-domain edges with the correct common commenter counts
inter_domain_edges_counts = {
    ('Influencer', 'Study'): 1175,
    ('Influencer', 'News'): 1042,
    ('Study', 'News'): 570
}

for domain_pair, weight in inter_domain_edges_counts.items():
    edge_color = inter_domain_edge_colors.get(domain_pair, 'black')
    combined_video_graph.add_edge(domain_pair[0], domain_pair[1], weight=weight, color=edge_color)

# Arrange domains in a triangular layout
domain_positions = {
    'Influencer': (0, 0),
    'News': (3, 0),  # Increase the distance for better visibility
    'Study': (1.5, np.sqrt(3))
}

# Arrange nodes in a circular layout within big circles
def arrange_in_circle(graph, domain_positions, domain, radius=1):
    nodes = [node for node in graph.nodes if isinstance(node, tuple) and node[1] == domain]
    num_nodes = len(nodes)
    angle_step = 2 * np.pi / num_nodes
    positions = {}
    for i, node in enumerate(sorted(nodes, key=lambda x: x[0])):  # Sort nodes for consistent ordering
        angle = i * angle_step
        positions[node] = (np.cos(angle) * radius + domain_positions[domain][0], 
                           np.sin(angle) * radius + domain_positions[domain][1])
    return positions

# Calculate positions for each domain's nodes
all_positions = {}
for domain in domain_positions:
    domain_pos = arrange_in_circle(combined_video_graph, domain_positions, domain)
    all_positions.update(domain_pos)

# For domain nodes, use the predefined domain positions
for node in domain_positions:
    all_positions[node] = domain_positions[node]

# Visualization
plt.figure(figsize=(20, 15))
edges = combined_video_graph.edges(data=True)

# Draw nodes with domain-based coloring
for node in combined_video_graph.nodes():
    nx.draw_networkx_nodes(combined_video_graph, all_positions, nodelist=[node], 
                           node_color=color_map[node[1]] if isinstance(node, tuple) else color_map[node], node_size=1000)

# Draw intra-domain edges
nx.draw_networkx_edges(combined_video_graph, all_positions, 
                       edgelist=[(u, v) for u, v, d in edges if u[1] == v[1]], 
                       width=1)

# Draw inter-domain edges with distinct colors
nx.draw_networkx_edges(combined_video_graph, all_positions, 
                       edgelist=[(u, v) for u, v, d in edges if u[1] != v[1]], 
                       edge_color='grey', style='dashed', width=2)

# Edge labels for inter-domain edges
nx.draw_networkx_edge_labels(combined_video_graph, all_positions, 
                             edge_labels={(u, v): d['weight'] for u, v, d in edges if u[1] != v[1]}, font_size=8)

# Domain labels
for domain, position in domain_positions.items():
    plt.text(position[0], position[1], domain, fontsize=15, ha='center')

plt.title("Video Interactions Within and Between Different Domains")
plt.axis("off")
plt.show()


# In[264]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Define file paths for the three Excel files
file_paths = {
    'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
    'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
    'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
}

# Load the data from each Excel file into dictionaries of DataFrames
domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# Create network graphs for each domain using 'Channel URL'
domain_graphs_url = {}
for domain, dfs in domain_dataframes.items():
    graph = nx.Graph()
    for sheet_name, df in dfs.items():
        if 'Video' in sheet_name and 'Channel URL' in df.columns:
            for other_sheet, other_df in dfs.items():
                if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
                    common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
                    if len(common_commenters) > 0:
                        graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
    domain_graphs_url[domain] = graph

# Collecting all commenters from each domain using 'Channel URL'
all_commenters_url = {}
for domain, dfs in domain_dataframes.items():
    domain_commenters = set()
    for df in dfs.values():
        if 'Channel URL' in df.columns:
            domain_commenters.update(df['Channel URL'].dropna())
    all_commenters_url[domain] = domain_commenters

# Calculate common commenters between each pair of domains
cross_domain_common_commenters = {}
for domain1, commenters1 in all_commenters_url.items():
    for domain2, commenters2 in all_commenters_url.items():
        if domain1 != domain2:
            common_commenters = commenters1.intersection(commenters2)
            domain_pair = tuple(sorted([domain1, domain2]))
            if domain_pair not in cross_domain_common_commenters:
                cross_domain_common_commenters[domain_pair] = len(common_commenters)

# Define specific colors for edges between each pair of domains
inter_domain_edge_colors = {
    ('Influencer', 'Study'): 'purple',
    ('Influencer', 'News'): 'orange',
    ('Study', 'News'): 'black'
}

# Define colors for each domain
color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# Create a combined graph for all videos from all domains
combined_video_graph = nx.Graph()

# Add intra-domain edges
for domain, graph in domain_graphs_url.items():
    for u, v, data in graph.edges(data=True):
        combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'], color=color_map[domain])

# Add inter-domain edges using the calculated common commenters
for domain_pair, common_count in cross_domain_common_commenters.items():
    edge_color = inter_domain_edge_colors.get(domain_pair, 'black')
    combined_video_graph.add_edge(domain_pair[0], domain_pair[1], weight=common_count, color=edge_color)

# Arrange domains in a triangular layout
domain_positions = {
    'Influencer': (0, 0),
    'News': (3, 0),  # Increase the distance for better visibility
    'Study': (1.5, np.sqrt(3))
}

# Arrange nodes in a circular layout within big circles
def arrange_in_circle(graph, domain_positions, domain, radius=1):
    nodes = [node for node in graph.nodes if isinstance(node, tuple) and node[1] == domain]
    num_nodes = len(nodes)
    angle_step = 2 * np.pi / num_nodes
    positions = {}
    for i, node in enumerate(sorted(nodes, key=lambda x: x[0])):  # Sort nodes for consistent ordering
        angle = i * angle_step
        positions[node] = (np.cos(angle) * radius + domain_positions[domain][0], 
                           np.sin(angle) * radius + domain_positions[domain][1])
    return positions

# Calculate positions for each domain's nodes
all_positions = {}
for domain in domain_positions:
    domain_pos = arrange_in_circle(combined_video_graph, domain_positions, domain)
    all_positions.update(domain_pos)

# For domain nodes, use the predefined domain positions
for node in domain_positions:
    all_positions[node] = domain_positions[node]

# Visualization
plt.figure(figsize=(20, 15))
edges = combined_video_graph.edges(data=True)

# Draw nodes with domain-based coloring
for node in combined_video_graph.nodes():
    nx.draw_networkx_nodes(combined_video_graph, all_positions, nodelist=[node], 
                           node_color=color_map[node[1]] if isinstance(node, tuple) else color_map[node], node_size=1000)

# Draw intra-domain edges
nx.draw_networkx_edges(combined_video_graph, all_positions, 
                       edgelist=[(u, v) for u, v, d in edges if u[1] == v[1]], 
                       width=1)

# Draw inter-domain edges with distinct colors
nx.draw_networkx_edges(combined_video_graph, all_positions, 
                       edgelist=[(u, v) for u, v, d in edges if u[1] != v[1]], 
                       edge_color='grey', style='dashed', width=2)

# Edge labels for inter-domain edges
nx.draw_networkx_edge_labels(combined_video_graph, all_positions, 
                             edge_labels={(u, v): d['weight'] for u, v, d in edges if u[1] != v[1]}, font_size=8)

# Domain labels
for domain, position in domain_positions.items():
    plt.text(position[0], position[1], domain, fontsize=15, ha='center')

plt.title("Video Interactions Within and Between Different Domains")
plt.axis("off")
plt.show()


# In[ ]:





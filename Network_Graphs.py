#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().system('pip install scipy')
get_ipython().system('pip install networkx')
get_ipython().system('pip install scipy==1.6.3')
get_ipython().system('pip install --upgrade networkx scipy')



# In[11]:


pip install python-louvain


# In[17]:


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

# Calculate node sizes and edge widths based on the graph's connectivity
degrees = dict(combined_video_graph.degree())
max_degree = max(degrees.values())
node_sizes = {node: (degrees[node] / max_degree) * 1000 + 100 for node in combined_video_graph.nodes()}  # The node size is scaled between 100 and 1100

# Generate positions for the nodes using a layout that respects weights
pos = nx.spring_layout(combined_video_graph, weight='weight', iterations=50)

# Visualization
plt.figure(figsize=(20, 15))

# Draw nodes with domain-based coloring and scaled sizes
nx.draw_networkx_nodes(
    combined_video_graph, pos,
    node_size=[node_sizes[node] for node in combined_video_graph.nodes()],
    node_color=[color_map.get(node[1], 'grey') for node in combined_video_graph.nodes()],
    alpha=0.9
)

# Draw edges with scaled widths
edge_weights = nx.get_edge_attributes(combined_video_graph, 'weight')
for (u, v, data) in combined_video_graph.edges(data=True):
    edge_color = data['color'] if 'color' in data else 'black'
    edge_width = (data['weight'] / max(edge_weights.values())) * 10
    nx.draw_networkx_edges(
        combined_video_graph, pos,
        edgelist=[(u, v)],
        width=edge_width,
        edge_color=edge_color
    )

# Draw labels
nx.draw_networkx_labels(
    combined_video_graph, pos,
    labels={node: node[0] for node in combined_video_graph.nodes()},
    font_size=8
)

plt.title("Common Commenters (Based on Channel URL) Within and Between Videos of Different Domains")
plt.axis('off')
plt.show()


# In[28]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community  # Ensure you have the 'python-louvain' package installed

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

# Create a combined graph for all videos from all domains
combined_video_graph = nx.Graph()

# Add intra-domain edges (within the same domain)
for domain, graph in domain_graphs_url.items():
    for u, v, data in graph.edges(data=True):
        combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'])

# Add inter-domain edges (between videos of different domains)
for video1, commenters1 in video_commenters.items():
    for video2, commenters2 in video_commenters.items():
        if video1[1] != video2[1]:  # Only for different domains
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                combined_video_graph.add_edge(video1, video2, weight=len(common_commenters))

# Use the Louvain community detection algorithm to find clusters
partition = community.best_partition(combined_video_graph, weight='weight')

# Calculate node sizes based on the number of common users
node_sizes = {
    node: 100 + 1900 * (len(commenters) / len(video_commenters[node]))
    for node, commenters in video_commenters.items()
}

# Generate positions for the nodes using a layout that respects weights
pos = nx.spring_layout(combined_video_graph, weight='weight', iterations=50)

# Visualization
plt.figure(figsize=(20, 15))

# Define unique colors for each domain
unique_domain_colors = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# Draw nodes with community-based coloring, scaled sizes, and 10 nodes of each color
node_colors = [unique_domain_colors[domain] for node, domain in combined_video_graph.nodes()]
nx.draw_networkx_nodes(
    combined_video_graph, pos,
    node_size=[node_sizes[node] for node in combined_video_graph.nodes()],
    node_color=node_colors,
    alpha=0.9
)

edge_weights = nx.get_edge_attributes(combined_video_graph, 'weight')
for (u, v, data) in combined_video_graph.edges(data=True):
    edge_width = (data['weight'] / max(edge_weights.values())) * 10
    nx.draw_networkx_edges(
        combined_video_graph, pos,
        edgelist=[(u, v)],
        width=edge_width,
        edge_color='grey',
        alpha=0.2
    )

# Draw labels
nx.draw_networkx_labels(
    combined_video_graph, pos,
    labels={node: node[0] for node in combined_video_graph.nodes()},
    font_size=8
)

plt.title("Common Commenters Clustering and Size based on Channel URL")
plt.axis('off')
plt.show()


# In[24]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community  # Ensure you have the 'python-louvain' package installed

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

# Create a combined graph for all videos from all domains
combined_video_graph = nx.Graph()

# Add intra-domain edges (within the same domain)
for domain, graph in domain_graphs_url.items():
    for u, v, data in graph.edges(data=True):
        combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'])

# Add inter-domain edges (between videos of different domains)
for video1, commenters1 in video_commenters.items():
    for video2, commenters2 in video_commenters.items():
        if video1[1] != video2[1]:  # Only for different domains
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                combined_video_graph.add_edge(video1, video2, weight=len(common_commenters))

# Use the Louvain community detection algorithm to find clusters
partition = community.best_partition(combined_video_graph, weight='weight')

# Calculate node sizes based on the degree centrality
node_sizes = nx.degree_centrality(combined_video_graph)
max_degree = max(node_sizes.values())
node_sizes = {k: 100 + 1900 * (v / max_degree) for k, v in node_sizes.items()}

# Generate positions for the nodes using a layout that respects weights
pos = nx.spring_layout(combined_video_graph, weight='weight', iterations=50)

# Visualization
plt.figure(figsize=(20, 15))

# Draw nodes with community-based coloring and scaled sizes
unique_cluster_colors = {0: 'lightblue', 1: 'lightgreen', 2: 'lightcoral'}  # Specify three unique colors for three clusters
node_colors = [unique_cluster_colors[partition[node]] for node in combined_video_graph.nodes()]

nx.draw_networkx_nodes(
    combined_video_graph, pos,
    node_size=[node_sizes[node] for node in combined_video_graph.nodes()],
    node_color=node_colors,
    alpha=0.9
)

edge_weights = nx.get_edge_attributes(combined_video_graph, 'weight')
for (u, v, data) in combined_video_graph.edges(data=True):
    edge_width = (data['weight'] / max(edge_weights.values())) * 10
    nx.draw_networkx_edges(
        combined_video_graph, pos,
        edgelist=[(u, v)],
        width=edge_width,
        edge_color='grey',
        alpha=0.2
    )

# Draw labels
nx.draw_networkx_labels(
    combined_video_graph, pos,
    labels={node: node[0] for node in combined_video_graph.nodes()},
    font_size=8
)

plt.title("Common Commenters Clustering and Size based on Channel URL")
plt.axis('off')
plt.show()


# In[27]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community  # Ensure you have the 'python-louvain' package installed

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

# Create a combined graph for all videos from all domains
combined_video_graph = nx.Graph()

# Add intra-domain edges (within the same domain)
for domain, graph in domain_graphs_url.items():
    for u, v, data in graph.edges(data=True):
        combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'])

# Add inter-domain edges (between videos of different domains)
for video1, commenters1 in video_commenters.items():
    for video2, commenters2 in video_commenters.items():
        if video1[1] != video2[1]:  # Only for different domains
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                combined_video_graph.add_edge(video1, video2, weight=len(common_commenters))

# Use the Louvain community detection algorithm to find clusters
partition = community.best_partition(combined_video_graph, weight='weight')

# Calculate node sizes based on the number of common users
node_sizes = {
    node: 100 + 1900 * (len(commenters) / len(video_commenters[node])) 
    for node, commenters in video_commenters.items()
}

# Generate positions for the nodes using a layout that respects weights
pos = nx.spring_layout(combined_video_graph, weight='weight', iterations=50)

# Visualization
plt.figure(figsize=(20, 15))

# Draw nodes with community-based coloring, scaled sizes, and larger circle borders
unique_cluster_colors = {0: 'lightblue', 1: 'lightgreen', 2: 'lightcoral'}  # Adjust as needed
node_colors = [unique_cluster_colors[partition[node]] for node in combined_video_graph.nodes()]

nx.draw_networkx_nodes(
    combined_video_graph, pos,
    node_size=[node_sizes[node] for node in combined_video_graph.nodes()],
    node_color=node_colors,
    alpha=0.9
)

# Draw larger circles around each cluster
for cluster, color in unique_cluster_colors.items():
    nodes_in_cluster = [node for node in combined_video_graph.nodes() if partition[node] == cluster]
    border_size = 300  # Adjust the border size as needed
    nx.draw_networkx_nodes(
        combined_video_graph, pos,
        nodelist=nodes_in_cluster,
        node_size=border_size,
        node_color='none',  # No fill color
        edgecolors=color,  # Cluster color for the border
        linewidths=3  # Border line width
    )

edge_weights = nx.get_edge_attributes(combined_video_graph, 'weight')
for (u, v, data) in combined_video_graph.edges(data=True):
    edge_width = (data['weight'] / max(edge_weights.values())) * 10
    nx.draw_networkx_edges(
        combined_video_graph, pos,
        edgelist=[(u, v)],
        width=edge_width,
        edge_color='grey',
        alpha=0.2
    )

# Draw labels
nx.draw_networkx_labels(
    combined_video_graph, pos,
    labels={node: node[0] for node in combined_video_graph.nodes()},
    font_size=8
)

plt.title("Common Commenters Clustering and Size based on Channel URL")
plt.axis('off')
plt.show()


# In[19]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community  # Ensure you have the 'python-louvain' package installed

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

# Create a combined graph for all videos from all domains
combined_video_graph = nx.Graph()

# Add intra-domain edges (within the same domain)
for domain, graph in domain_graphs_url.items():
    for u, v, data in graph.edges(data=True):
        combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'])

# Add inter-domain edges (between videos of different domains)
for video1, commenters1 in video_commenters.items():
    for video2, commenters2 in video_commenters.items():
        if video1[1] != video2[1]:  # Only for different domains
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                combined_video_graph.add_edge(video1, video2, weight=len(common_commenters))

# Use the Louvain community detection algorithm to find clusters
partition = community.best_partition(combined_video_graph, weight='weight')

# Calculate node sizes based on the number of common users
node_sizes = {
    node: 100 + 1900 * (len(commenters) / len(video_commenters[node])) 
    for node, commenters in video_commenters.items()
}

# Generate positions for the nodes using a layout that respects weights
pos = nx.spring_layout(combined_video_graph, weight='weight', iterations=50)

# Visualization
plt.figure(figsize=(20, 15))

# Draw nodes with domain-based coloring and updated sizes
node_colors = [partition[node] for node in combined_video_graph.nodes()]
domain_colors = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}
nx.draw_networkx_nodes(
    combined_video_graph, pos,
    node_size=[node_sizes[node] for node in combined_video_graph.nodes()],
    node_color=[domain_colors[node[1]] for node in combined_video_graph.nodes()],
    cmap=plt.cm.get_cmap('viridis'),
    alpha=0.9
)

# Draw edges with scaled widths
edge_weights = nx.get_edge_attributes(combined_video_graph, 'weight')
for (u, v, data) in combined_video_graph.edges(data=True):
    edge_width = (data['weight'] / max(edge_weights.values())) * 10
    nx.draw_networkx_edges(
        combined_video_graph, pos,
        edgelist=[(u, v)],
        width=edge_width,
        edge_color='grey',
        alpha=0.2
    )

# Draw labels
nx.draw_networkx_labels(
    combined_video_graph, pos,
    labels={node: node[0] for node in combined_video_graph.nodes()},
    font_size=8
)

plt.title("Common Commenters Clustering and Size based on Channel URL")
plt.axis('off')
plt.show()


# In[20]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community  # Ensure you have the 'python-louvain' package installed

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

# Create a combined graph for all videos from all domains
combined_video_graph = nx.Graph()

# Add intra-domain edges (within the same domain)
for domain, graph in domain_graphs_url.items():
    for u, v, data in graph.edges(data=True):
        combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'])

# Add inter-domain edges (between videos of different domains)
for video1, commenters1 in video_commenters.items():
    for video2, commenters2 in video_commenters.items():
        if video1[1] != video2[1]:  # Only for different domains
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                combined_video_graph.add_edge(video1, video2, weight=len(common_commenters))

# Use the Louvain community detection algorithm to find clusters
partition = community.best_partition(combined_video_graph, weight='weight')

# Calculate node sizes based on the number of common users
degrees = dict(combined_video_graph.degree())
max_degree = max(degrees.values())
node_sizes = {
    node: 100 + 1900 * (degrees[node] / max_degree)
    for node in combined_video_graph.nodes()
}

# Generate positions for the nodes using a layout that respects weights
pos = nx.spring_layout(combined_video_graph, weight='weight', iterations=50)

# Visualization
plt.figure(figsize=(20, 15))

# Draw nodes with domain-based coloring and updated sizes
node_colors = [partition[node] for node in combined_video_graph.nodes()]
domain_colors = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}
nx.draw_networkx_nodes(
    combined_video_graph, pos,
    node_size=[node_sizes[node] for node in combined_video_graph.nodes()],
    node_color=[domain_colors[node[1]] for node in combined_video_graph.nodes()],
    cmap=plt.cm.get_cmap('viridis'),
    alpha=0.9
)

# Draw edges with scaled widths
edge_weights = nx.get_edge_attributes(combined_video_graph, 'weight')
for (u, v, data) in combined_video_graph.edges(data=True):
    edge_width = (data['weight'] / max(edge_weights.values())) * 10
    nx.draw_networkx_edges(
        combined_video_graph, pos,
        edgelist=[(u, v)],
        width=edge_width,
        edge_color='grey',
        alpha=0.2
    )

# Draw labels
nx.draw_networkx_labels(
    combined_video_graph, pos,
    labels={node: node[0] for node in combined_video_graph.nodes()},
    font_size=8
)

plt.title("Common Commenters Clustering and Size based on Channel URL")
plt.axis('off')
plt.show()


# In[21]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community  # Ensure you have the 'python-louvain' package installed

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

# Create a combined graph for all videos from all domains
combined_video_graph = nx.Graph()

# Add intra-domain edges (within the same domain)
for domain, graph in domain_graphs_url.items():
    for u, v, data in graph.edges(data=True):
        combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'])

# Add inter-domain edges (between videos of different domains)
for video1, commenters1 in video_commenters.items():
    for video2, commenters2 in video_commenters.items():
        if video1[1] != video2[1]:  # Only for different domains
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                combined_video_graph.add_edge(video1, video2, weight=len(common_commenters))

# Use the Louvain community detection algorithm to find clusters
partition = community.best_partition(combined_video_graph, weight='weight')

# Calculate node sizes based on the number of common users
node_sizes = {
    node: 100 + 1900 * (len(commenters) / len(video_commenters[node]))
    for node, commenters in video_commenters.items()
}

# Generate positions for the nodes using a layout that respects weights
pos = nx.spring_layout(combined_video_graph, weight='weight', iterations=50)

# Visualization
plt.figure(figsize=(20, 15))

# Draw nodes with community-based coloring and scaled sizes
node_colors = [partition[node] for node in combined_video_graph.nodes()]
domain_colors = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

nx.draw_networkx_nodes(
    combined_video_graph, pos,
    node_size=[node_sizes[node] for node in combined_video_graph.nodes()],
    node_color=[domain_colors[node[1]] for node in combined_video_graph.nodes()],
    cmap=plt.cm.get_cmap('viridis'),
    alpha=0.9
)

# Draw edges with scaled widths
edge_weights = nx.get_edge_attributes(combined_video_graph, 'weight')
for (u, v, data) in combined_video_graph.edges(data=True):
    edge_width = (data['weight'] / max(edge_weights.values())) * 10
    nx.draw_networkx_edges(
        combined_video_graph, pos,
        edgelist=[(u, v)],
        width=edge_width,
        edge_color='grey',
        alpha=0.2
    )

# Draw labels
nx.draw_networkx_labels(
    combined_video_graph, pos,
    labels={node: node[0] for node in combined_video_graph.nodes()},
    font_size=8
)

plt.title("Common Commenters Clustering and Size based on Channel URL")
plt.axis('off')
plt.show()


# In[2]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
z
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


# In[3]:


#News channels


# In[4]:


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


# In[5]:


#Study Channel


# In[6]:


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


# In[7]:


# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt

# # Define file paths for the three Excel files
# file_paths = {
#     'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
#     'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
#     'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
# }

# # Load the data from each Excel file into dictionaries of DataFrames
# domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# # Create network graphs for each domain
# domain_graphs = {}
# for domain, dfs in domain_dataframes.items():
#     graph = nx.Graph()
#     for sheet_name, df in dfs.items():
#         if 'Video' in sheet_name:  # Filter out irrelevant sheets
#             for other_sheet, other_df in dfs.items():
#                 if 'Video' in other_sheet and sheet_name != other_sheet:
#                     # Find common commenters between sheets
#                     common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
#                     if len(common_commenters) > 0:
#                         # Add an edge for common commenters
#                         graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
#     domain_graphs[domain] = graph

# # Visualization
# plt.figure(figsize=(20, 15))

# # Define colors for each domain
# color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# # Draw the graphs for each domain with distinct colors
# for domain, graph in domain_graphs.items():
#     pos = nx.spring_layout(graph, seed=42)  # Consistent layout for all domains
#     nx.draw_networkx(graph, pos, with_labels=True, node_color=color_map[domain], 
#                      node_size=3000, edge_color='gray', width=2, font_size=12)

# plt.title("Common Commenters Across Different Domains")
# plt.axis("off")
# plt.show()


# In[8]:


# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt

# # Define file paths for the three Excel files
# file_paths = {
#     'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
#     'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
#     'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
# }

# # Load the data from each Excel file into dictionaries of DataFrames
# domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# # Create network graphs for each domain
# domain_graphs = {}
# for domain, dfs in domain_dataframes.items():
#     graph = nx.Graph()
#     for sheet_name, df in dfs.items():
#         if 'Video' in sheet_name:  # Filter out irrelevant sheets
#             for other_sheet, other_df in dfs.items():
#                 if 'Video' in other_sheet and sheet_name != other_sheet:
#                     # Find common commenters between sheets
#                     common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
#                     if len(common_commenters) > 0:
#                         # Add an edge for common commenters
#                         graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
#     domain_graphs[domain] = graph

# # Visualization with subplots
# fig, axs = plt.subplots(1, 3, figsize=(20, 7))  # 1 row, 3 columns

# # Define colors for each domain
# color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# # Draw the graphs for each domain in separate subplots
# for (domain, graph), ax in zip(domain_graphs.items(), axs):
#     pos = nx.spring_layout(graph, seed=42)  # Consistent layout for all domains
#     nx.draw_networkx(graph, pos, ax=ax, with_labels=True, node_color=color_map[domain], 
#                      node_size=1000, edge_color='gray', width=1, font_size=10)
#     ax.set_title(f"{domain} Domain")
#     ax.axis("off")

# plt.suptitle("Common Commenters Across Different Domains")
# plt.tight_layout()
# plt.show()


# In[9]:


# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt

# # Define file paths for the three Excel files
# file_paths = {
#     'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
#     'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
#     'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
# }


# # Load the data from each Excel file into dictionaries of DataFrames
# domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# # Create network graphs for each domain
# domain_graphs = {}
# for domain, dfs in domain_dataframes.items():
#     graph = nx.Graph()
#     for sheet_name, df in dfs.items():
#         if 'Video' in sheet_name:  # Filter out irrelevant sheets
#             for other_sheet, other_df in dfs.items():
#                 if 'Video' in other_sheet and sheet_name != other_sheet:
#                     # Find common commenters between sheets
#                     common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
#                     if len(common_commenters) > 0:
#                         # Add an edge for common commenters
#                         graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
#     domain_graphs[domain] = graph

# # Visualization with subplots and edge weights
# fig, axs = plt.subplots(1, 3, figsize=(20, 7))  # 1 row, 3 columns

# # Define colors for each domain
# color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# # Draw the graphs for each domain in separate subplots with edge labels
# for (domain, graph), ax in zip(domain_graphs.items(), axs):
#     pos = nx.spring_layout(graph, seed=42)  # Consistent layout for all domains
#     nx.draw_networkx(graph, pos, ax=ax, with_labels=True, node_color=color_map[domain], 
#                      node_size=1000, edge_color='gray', width=1, font_size=10)
    
#     # Adding edge labels for weights
#     edge_weights = nx.get_edge_attributes(graph, 'weight')
#     nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, font_size=8, ax=ax)

#     ax.set_title(f"{domain} Domain")
#     ax.axis("off")

# plt.suptitle("Common Commenters Across Different Domains")
# plt.tight_layout()
# plt.show()


# In[10]:


# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt

# # Define file paths for the three Excel files
# file_paths = {
#     'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
#     'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
#     'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
# }


# # Load the data from each Excel file into dictionaries of DataFrames
# domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# # Create network graphs for each domain
# domain_graphs = {}
# for domain, dfs in domain_dataframes.items():
#     graph = nx.Graph()
#     for sheet_name, df in dfs.items():
#         if 'Video' in sheet_name and 'Channel URL' in df.columns:  # Filter out irrelevant sheets and check 'Name' column
#             for other_sheet, other_df in dfs.items():
#                 if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
#                     # Find common commenters between sheets
#                     common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
#                     if len(common_commenters) > 0:
#                         # Add an edge for common commenters
#                         graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
#     domain_graphs[domain] = graph

# # Collecting all commenters from each domain
# all_commenters = {}
# for domain, dfs in domain_dataframes.items():
#     domain_commenters = set()
#     for df in dfs.values():
#         if 'Channel URL' in df.columns:  # Check if 'Name' column is present
#             domain_commenters.update(df['Channel URL'].dropna())
#     all_commenters[domain] = domain_commenters

# # Create a separate graph for cross-domain connections
# cross_domain_graph = nx.Graph()

# # Define colors for each domain and cross-domain edges
# color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}
# cross_domain_edge_color = 'red'

# # Add cross-domain edges
# for domain1, commenters1 in all_commenters.items():
#     for domain2, commenters2 in all_commenters.items():
#         if domain1 != domain2:
#             common_commenters = commenters1.intersection(commenters2)
#             if len(common_commenters) > 0:
#                 cross_domain_graph.add_edge(domain1, domain2, weight=len(common_commenters))

# # Visualization with subplots for intra-domain and a separate subplot for cross-domain connections
# fig, axs = plt.subplots(1, 4, figsize=(25, 7))  # 1 row, 4 columns

# # Draw the intra-domain graphs in the first three subplots
# for (domain, graph), ax in zip(domain_graphs.items(), axs[:3]):
#     pos = nx.spring_layout(graph, seed=42)  # Consistent layout for all domains
#     nx.draw_networkx(graph, pos, ax=ax, with_labels=True, node_color=color_map[domain], 
#                      node_size=1000, edge_color='gray', width=1, font_size=10)
    
#     # Adding edge labels for weights within domains
#     edge_weights = nx.get_edge_attributes(graph, 'weight')
#     nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, font_size=8, ax=ax)

#     ax.set_title(f"{domain} Domain")
#     ax.axis("off")

# # Draw the cross-domain graph in the fourth subplot
# ax_cross_domain = axs[3]
# pos_cross_domain = nx.spring_layout(cross_domain_graph, seed=42)
# nx.draw_networkx(cross_domain_graph, pos_cross_domain, ax=ax_cross_domain, with_labels=True, 
#                  node_size=1000, edge_color=cross_domain_edge_color, width=1, font_size=10)

# # Adding edge labels for weights between domains
# edge_weights_cross_domain = nx.get_edge_attributes(cross_domain_graph, 'weight')
# nx.draw_networkx_edge_labels(cross_domain_graph, pos_cross_domain, edge_labels=edge_weights_cross_domain, font_size=8, ax=ax_cross_domain)

# ax_cross_domain.set_title("Cross-Domain Connections")
# ax_cross_domain.axis("off")

# plt.suptitle("Common Commenters Within and Between Different Domains")
# plt.tight_layout()
# plt.show()


# In[11]:


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


# In[12]:


# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt

# # Define file paths for the three Excel files
# file_paths = {
#     'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
#     'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
#     'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
# }

# # Load the data from each Excel file into dictionaries of DataFrames
# domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# # Create network graphs for each domain using 'Channel URL'
# domain_graphs_url = {}
# for domain, dfs in domain_dataframes.items():
#     graph = nx.Graph()
#     for sheet_name, df in dfs.items():
#         if 'Video' in sheet_name and 'Channel URL' in df.columns:  # Check for the right column
#             for other_sheet, other_df in dfs.items():
#                 if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
#                     # Find common commenters using 'Channel URL'
#                     common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
#                     if len(common_commenters) > 0:
#                         graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
#     domain_graphs_url[domain] = graph

# # Collecting all commenters for each video across all domains using 'Channel URL'
# video_commenters = {}
# for domain, dfs in domain_dataframes.items():
#     for video, df in dfs.items():
#         if 'Video' in video and 'Channel URL' in df.columns:
#             video_commenters[(video, domain)] = set(df['Channel URL'].dropna())

# # Create a combined graph for all videos from all domains
# combined_video_graph = nx.Graph()

# # Define colors for each domain and inter-domain edges
# color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}
# inter_domain_edge_color = 'purple'

# # Add intra-domain edges (within the same domain)
# for domain, graph in domain_graphs_url.items():
#     for u, v, data in graph.edges(data=True):
#         combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'], color=color_map[domain])

# # Add inter-domain edges (between videos of different domains)
# for video1, commenters1 in video_commenters.items():
#     for video2, commenters2 in video_commenters.items():
#         if video1[1] != video2[1]:  # Ensure videos are from different domains
#             common_commenters = commenters1.intersection(commenters2)
#             if len(common_commenters) > 0:
#                 combined_video_graph.add_edge(video1, video2, weight=len(common_commenters), color=inter_domain_edge_color)

# # Visualization with a single graph
# plt.figure(figsize=(20, 15))
# pos = nx.spring_layout(combined_video_graph, seed=42)  # Layout for visual clarity
# edges = combined_video_graph.edges(data=True)

# # Draw nodes with correct domain-based coloring
# nx.draw_networkx_nodes(combined_video_graph, pos, node_size=1000, 
#                        node_color=[color_map[node[1]] for node in combined_video_graph.nodes()])

# # Draw intra-domain edges
# nx.draw_networkx_edges(combined_video_graph, pos, 
#                        edgelist=[(u, v) for u, v, d in edges if 'color' in d and d['color'] != inter_domain_edge_color], 
#                        width=1)

# # Draw inter-domain edges
# nx.draw_networkx_edges(combined_video_graph, pos, 
#                        edgelist=[(u, v) for u, v, d in edges if 'color' in d and d['color'] == inter_domain_edge_color], 
#                        edge_color=inter_domain_edge_color, style='dashed', width=1)

# # Edge labels
# nx.draw_networkx_edge_labels(combined_video_graph, pos, 
#                              edge_labels={(u, v): d['weight'] for u, v, d in edges}, font_size=8)

# # Node labels
# nx.draw_networkx_labels(combined_video_graph, pos, 
#                         labels={n: n[0] for n in combined_video_graph.nodes()}, font_size=10)

# plt.title("Common Commenters (Based on Channel URL) Within and Between Videos of Different Domains")
# plt.axis("off")
# plt.show()


# In[13]:


# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt

# # Define file paths for the three Excel files
# file_paths = {
#     'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
#     'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
#     'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
# }

# # Load the data from each Excel file into dictionaries of DataFrames
# domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# # Create network graphs for each domain using 'Channel URL'
# domain_graphs_url = {}
# for domain, dfs in domain_dataframes.items():
#     graph = nx.Graph()
#     for sheet_name, df in dfs.items():
#         if 'Video' in sheet_name and 'Channel URL' in df.columns:
#             for other_sheet, other_df in dfs.items():
#                 if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
#                     common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
#                     if len(common_commenters) > 0:
#                         graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
#     domain_graphs_url[domain] = graph

# # Collecting all commenters for each video across all domains using 'Channel URL'
# video_commenters = {}
# for domain, dfs in domain_dataframes.items():
#     for video, df in dfs.items():
#         if 'Video' in video and 'Channel URL' in df.columns:
#             video_commenters[(video, domain)] = set(df['Channel URL'].dropna())

# # Create a combined graph for all videos from all domains
# combined_video_graph = nx.Graph()

# # Define colors for each domain and inter-domain edges
# color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}
# inter_domain_edge_color = 'purple'

# # Add intra-domain edges (within the same domain)
# for domain, graph in domain_graphs_url.items():
#     for u, v, data in graph.edges(data=True):
#         combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'], color=color_map[domain])

# # Add inter-domain edges (between videos of different domains)
# for video1, commenters1 in video_commenters.items():
#     for video2, commenters2 in video_commenters.items():
#         if video1[1] != video2[1]:
#             common_commenters = commenters1.intersection(commenters2)
#             if len(common_commenters) > 0:
#                 combined_video_graph.add_edge(video1, video2, weight=len(common_commenters), color=inter_domain_edge_color)

# # Function to calculate positions for each domain in a triangular layout
# def calculate_domain_positions(graph, domain, offset, layout_func=nx.spring_layout):
#     domain_nodes = [node for node in graph.nodes if node[1] == domain]
#     subgraph = graph.subgraph(domain_nodes)
#     pos = layout_func(subgraph, seed=42)
#     pos = {node: (x + offset[0], y + offset[1]) for node, (x, y) in pos.items()}
#     return pos

# # Prepare a combined position dictionary for all nodes
# combined_pos = {}

# # Calculate positions for each domain with an offset to arrange them in a triangular layout
# offsets = {'Influencer': (0, 0), 'News': (1, 1), 'Study': (2, 0)}
# for domain in domain_graphs_url.keys():
#     domain_pos = calculate_domain_positions(combined_video_graph, domain, offsets[domain])
#     combined_pos.update(domain_pos)

# # Visualization with triangular layout
# plt.figure(figsize=(20, 15))
# edges = combined_video_graph.edges(data=True)

# # Draw nodes with domain-based coloring
# nx.draw_networkx_nodes(combined_video_graph, combined_pos, node_size=1000, 
#                        node_color=[color_map[node[1]] for node in combined_video_graph.nodes()])

# # Draw intra-domain edges
# nx.draw_networkx_edges(combined_video_graph, combined_pos, 
#                        edgelist=[(u, v) for u, v, d in edges if 'color' in d and d['color'] != inter_domain_edge_color], 
#                        width=1)

# # Draw inter-domain edges
# nx.draw_networkx_edges(combined_video_graph, combined_pos, 
#                        edgelist=[(u, v) for u, v, d in edges if 'color' in d and d['color'] == inter_domain_edge_color], 
#                        edge_color=inter_domain_edge_color, style='dashed', width=1)

# # Edge labels
# nx.draw_networkx_edge_labels(combined_video_graph, combined_pos, 
#                              edge_labels={(u, v): d['weight'] for u, v, d in edges}, font_size=8)

# # Node labels
# nx.draw_networkx_labels(combined_video_graph, combined_pos, 
#                         labels={n: n[0] for n in combined_video_graph.nodes()}, font_size=10)

# plt.title("Common Commenters (Based on Channel URL) Within and Between Videos of Different Domains")
# plt.axis("off")
# plt.show()


# In[14]:


# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt

# # Load the data from each Excel file into dictionaries of DataFrames
# file_paths = {
#     'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
#     'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
#     'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
# }
# domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# # Create network graphs for each domain using 'Channel URL'
# domain_graphs_url = {}
# for domain, dfs in domain_dataframes.items():
#     graph = nx.Graph()
#     for sheet_name, df in dfs.items():
#         if 'Video' in sheet_name and 'Channel URL' in df.columns:
#             for other_sheet, other_df in dfs.items():
#                 if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
#                     common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
#                     if len(common_commenters) > 0:
#                         graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
#     domain_graphs_url[domain] = graph

# # Collecting all commenters for each video across all domains using 'Channel URL'
# video_commenters = {}
# for domain, dfs in domain_dataframes.items():
#     for video, df in dfs.items():
#         if 'Video' in video and 'Channel URL' in df.columns:
#             video_commenters[(video, domain)] = set(df['Channel URL'].dropna())

# # Define specific colors for edges between each pair of domains
# inter_domain_edge_colors = {
#     ('Influencer', 'Study'): 'purple',
#     ('Influencer', 'News'): 'orange',
#     ('Study', 'News'): 'green'
# }

# # Define colors for each domain
# color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# # Create a combined graph for all videos from all domains
# combined_video_graph = nx.Graph()

# # Add intra-domain edges (within the same domain)
# for domain, graph in domain_graphs_url.items():
#     for u, v, data in graph.edges(data=True):
#         combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'], color=color_map[domain])

# # Add inter-domain edges (between videos of different domains)
# for video1, commenters1 in video_commenters.items():
#     for video2, commenters2 in video_commenters.items():
#         if video1[1] != video2[1]:  # Only for different domains
#             common_commenters = commenters1.intersection(commenters2)
#             if len(common_commenters) > 0:
#                 # Determine the color based on the pair of domains
#                 domain_pair = tuple(sorted([video1[1], video2[1]]))
#                 edge_color = inter_domain_edge_colors.get(domain_pair, 'black')  # Default to black if pair not defined
#                 combined_video_graph.add_edge(video1, video2, weight=len(common_commenters), color=edge_color)

# # Visualization with triangular layout and distinct inter-domain edge colors
# plt.figure(figsize=(20, 15))
# edges = combined_video_graph.edges(data=True)

# # Draw nodes with domain-based coloring
# nx.draw_networkx_nodes(combined_video_graph, pos, node_size=1000, 
#                        node_color=[color_map[node[1]] for node in combined_video_graph.nodes()])

# # Draw intra-domain edges
# nx.draw_networkx_edges(combined_video_graph, pos, 
#                        edgelist=[(u, v) for u, v, d in edges if 'color' in d and d['color'] in color_map.values()], 
#                        width=1)

# # Draw inter-domain edges with distinct colors
# for domain_pair, edge_color in inter_domain_edge_colors.items():
#     nx.draw_networkx_edges(combined_video_graph, pos, 
#                            edgelist=[(u, v) for u, v, d in edges if 'color' in d and d['color'] == edge_color], 
#                            edge_color=edge_color, style='dashed', width=1)

# # Edge labels
# nx.draw_networkx_edge_labels(combined_video_graph, pos, 
#                              edge_labels={(u, v): d['weight'] for u, v, d in edges}, font_size=8)

# # Node labels
# nx.draw_networkx_labels(combined_video_graph, pos, 
#                         labels={n: n[0] for n in combined_video_graph.nodes()}, font_size=10)

# plt.title("Common Commenters (Based on Channel URL) Within and Between Videos of Different Domains")
# plt.axis("off")
# plt.show()


# In[15]:


# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np

# # Define file paths for the three Excel files
# file_paths = {
#     'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
#     'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
#     'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
# }

# # Load the data from each Excel file into dictionaries of DataFrames
# domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# # Create network graphs for each domain using 'Channel URL'
# domain_graphs_url = {}
# for domain, dfs in domain_dataframes.items():
#     graph = nx.Graph()
#     for sheet_name, df in dfs.items():
#         if 'Video' in sheet_name and 'Channel URL' in df.columns:
#             for other_sheet, other_df in dfs.items():
#                 if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
#                     common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
#                     if len(common_commenters) > 0:
#                         graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
#     domain_graphs_url[domain] = graph

# # Collecting all commenters for each video across all domains using 'Channel URL'
# video_commenters = {}
# for domain, dfs in domain_dataframes.items():
#     for video, df in dfs.items():
#         if 'Video' in video and 'Channel URL' in df.columns:
#             video_commenters[(video, domain)] = set(df['Channel URL'].dropna())

# # Define specific colors for edges between each pair of domains
# inter_domain_edge_colors = {
#     ('Influencer', 'Study'): 'purple',
#     ('Influencer', 'News'): 'orange',
#     ('Study', 'News'): 'black'  # Ensuring Study-News edges are orange
# }

# # Define colors for each domain
# color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# # Create a combined graph for all videos from all domains
# combined_video_graph = nx.Graph()

# # Add intra-domain edges (within the same domain)
# for domain, graph in domain_graphs_url.items():
#     for u, v, data in graph.edges(data=True):
#         combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'], color=color_map[domain])

# # Add inter-domain edges (between videos of different domains)
# for video1, commenters1 in video_commenters.items():
#     for video2, commenters2 in video_commenters.items():
#         if video1[1] != video2[1]:  # Only for different domains
#             common_commenters = commenters1.intersection(commenters2)
#             if len(common_commenters) > 0:
#                 # Sort the domain pair to match the keys in the inter_domain_edge_colors
#                 domain_pair = tuple(sorted([video1[1], video2[1]]))
#                 # Use the get method to provide a default color ('black') if the key doesn't exist
#                 edge_color = inter_domain_edge_colors.get(domain_pair, 'black')
#                 combined_video_graph.add_edge(video1, video2, weight=len(common_commenters), color=edge_color)

# # Function to arrange nodes of each domain in a triangular layout
# def arrange_in_triangle(graph, domain_positions, domain):
#     nodes = [node for node in graph.nodes if node[1] == domain]
#     num_nodes = len(nodes)
#     angle_step = np.pi / 2 / (num_nodes - 1)  # Spread nodes evenly
#     positions = {}
#     start_angle = np.pi / 4
#     radius = 1.5  # Increase radius for better spread
#     for i, node in enumerate(sorted(nodes)):  # Sort nodes for consistent ordering
#         angle = start_angle + i * angle_step
#         positions[node] = (np.cos(angle) * radius + domain_positions[domain][0], 
#                            np.sin(angle) * radius + domain_positions[domain][1])
#     return positions

# # Vertices of an equilateral triangle for the three domains
# domain_vertices = {
#     'Influencer':(0,0),
#     'News':(1,np.sqrt(3)/2),
#     'Study': (2, 0)
# }

# # Calculate positions for each domain's nodes
# all_positions = {}
# for domain in domain_vertices:
#     domain_pos = arrange_in_triangle(combined_video_graph, domain_vertices, domain)
#     all_positions.update(domain_pos)

# # Visualization with triangular layout and distinct inter-domain edge colors
# plt.figure(figsize=(20, 15))
# edges = combined_video_graph.edges(data=True)

# # Draw nodes with domain-based coloring
# nx.draw_networkx_nodes(combined_video_graph, all_positions, node_size=1000, 
#                        node_color=[color_map[node[1]] for node in combined_video_graph.nodes()])

# # Draw intra-domain edges
# nx.draw_networkx_edges(combined_video_graph, all_positions, 
#                        edgelist=[(u, v) for u, v, d in edges if d['color'] in color_map.values()], 
#                        width=1)

# # Draw all inter-domain edges with distinct colors for each pair of domains
# for domain_pair, edge_color in inter_domain_edge_colors.items():
#     nx.draw_networkx_edges(combined_video_graph, all_positions, 
#                            edgelist=[(u, v) for u, v, d in edges if d['color'] == edge_color], 
#                            edge_color=edge_color, style='dashed', width=2)

# # Edge labels
# nx.draw_networkx_edge_labels(combined_video_graph, all_positions, 
#                              edge_labels={(u, v): d['weight'] for u, v, d in edges}, font_size=8)

# # Node labels
# nx.draw_networkx_labels(combined_video_graph, all_positions, 
#                         labels={n: n[0] for n in combined_video_graph.nodes()}, font_size=10)

# plt.title("Common Commenters (Based on Channel URL) Within and Between Videos of Different Domains")
# plt.axis("off")
# plt.show()


# In[16]:


# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np

# # Define file paths for the three Excel files
# file_paths = {
#     'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
#     'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
#     'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
# }

# # Load the data from each Excel file into dictionaries of DataFrames
# domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# # Create network graphs for each domain using 'Channel URL'
# domain_graphs_url = {}
# for domain, dfs in domain_dataframes.items():
#     graph = nx.Graph()
#     for sheet_name, df in dfs.items():
#         if 'Video' in sheet_name and 'Channel URL' in df.columns:
#             for other_sheet, other_df in dfs.items():
#                 if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
#                     common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
#                     if len(common_commenters) > 0:
#                         graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
#     domain_graphs_url[domain] = graph

# # Collecting all commenters for each video across all domains using 'Channel URL'
# video_commenters = {}
# for domain, dfs in domain_dataframes.items():
#     for video, df in dfs.items():
#         if 'Video' in video and 'Channel URL' in df.columns:
#             video_commenters[(video, domain)] = set(df['Channel URL'].dropna())

# # Define specific colors for edges between each pair of domains
# inter_domain_edge_colors = {
#     ('Influencer', 'Study'): 'purple',
#     ('Influencer', 'News'): 'orange',
#     ('Study', 'News'): 'black'
# }

# # Define colors for each domain
# color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# # Create a combined graph for all videos from all domains
# combined_video_graph = nx.Graph()

# # Add intra-domain edges
# for domain, graph in domain_graphs_url.items():
#     for u, v, data in graph.edges(data=True):
#         combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'], color=color_map[domain])

# # Add inter-domain edges
# for video1, commenters1 in video_commenters.items():
#     for video2, commenters2 in video_commenters.items():
#         if video1[1] != video2[1]:  # Only for different domains
#             common_commenters = commenters1.intersection(commenters2)
#             if len(common_commenters) > 0:
#                 domain_pair = tuple(sorted([video1[1], video2[1]]))
#                 edge_color = inter_domain_edge_colors.get(domain_pair, 'black')
#                 combined_video_graph.add_edge(video1, video2, weight=len(common_commenters), color=edge_color)

# # Arrange nodes in a circular layout within big circles
# def arrange_in_circle(graph, domain_positions, domain, radius=1):
#     nodes = [node for node in graph.nodes if node[1] == domain]
#     num_nodes = len(nodes)
#     angle_step = 2 * np.pi / num_nodes
#     positions = {}
#     for i, node in enumerate(nodes):
#         angle = i * angle_step
#         positions[node] = (np.cos(angle) * radius + domain_positions[domain][0], 
#                            np.sin(angle) * radius + domain_positions[domain][1])
#     return positions

# # Big circle centers for the three domains
# domain_centers = {
#     'Influencer': (0, 0),
#     'News': (4, 0),
#     'Study': (8, 0)
# }

# # Calculate positions for each domain's nodes
# all_positions = {}
# for domain in domain_centers:
#     domain_pos = arrange_in_circle(combined_video_graph, domain_centers, domain)
#     all_positions.update(domain_pos)

# # Visualization
# plt.figure(figsize=(20, 15))
# edges = combined_video_graph.edges(data=True)

# # Draw big circles for each domain
# for domain, center in domain_centers.items():
#     circle = plt.Circle(center, 1, color=color_map[domain], fill=False)
#     plt.gca().add_patch(circle)

# # Draw nodes with domain-based coloring
# nx.draw_networkx_nodes(combined_video_graph, all_positions, node_size=1000, 
#                        node_color=[color_map[node[1]] for node in combined_video_graph.nodes()])

# # Draw intra-domain edges
# nx.draw_networkx_edges(combined_video_graph, all_positions, 
#                        edgelist=[(u, v) for u, v, d in edges if d['color'] in color_map.values()], 
#                        width=1)

# # Draw inter-domain edges with distinct colors
# for domain_pair, edge_color in inter_domain_edge_colors.items():
#     nx.draw_networkx_edges(combined_video_graph, all_positions, 
#                            edgelist=[(u, v) for u, v, d in edges if d['color'] == edge_color], 
#                            edge_color=edge_color, style='dashed', width=2)

# # Edge labels
# nx.draw_networkx_edge_labels(combined_video_graph, all_positions, 
#                              edge_labels={(u, v): d['weight'] for u, v, d in edges}, font_size=8)

# # Node labels
# nx.draw_networkx_labels(combined_video_graph, all_positions, 
#                         labels={n: n[0] for n in combined_video_graph.nodes()}, font_size=10)

# plt.title("Common Commenters Within and Between Videos of Different Domains")
# plt.axis("off")
# plt.show()


# In[17]:


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


# In[18]:


# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np

# # Define file paths for the three Excel files
# file_paths = {
#     'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
#     'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
#     'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
# }

# # Load the data from each Excel file into dictionaries of DataFrames
# domain_dataframes = {domain: pd.read_excel(file_path, sheet_name=None) for domain, file_path in file_paths.items()}

# # Create network graphs for each domain using 'Channel URL'
# domain_graphs_url = {}
# for domain, dfs in domain_dataframes.items():
#     graph = nx.Graph()
#     for sheet_name, df in dfs.items():
#         if 'Video' in sheet_name and 'Channel URL' in df.columns:
#             for other_sheet, other_df in dfs.items():
#                 if 'Video' in other_sheet and 'Channel URL' in other_df.columns and sheet_name != other_sheet:
#                     common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
#                     if len(common_commenters) > 0:
#                         graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
#     domain_graphs_url[domain] = graph

# # Collecting all commenters for each video across all domains using 'Channel URL'
# video_commenters = {}
# for domain, dfs in domain_dataframes.items():
#     for video, df in dfs.items():
#         if 'Video' in video and 'Channel URL' in df.columns:
#             video_commenters[(video, domain)] = set(df['Channel URL'].dropna())

# # Define specific colors for edges between each pair of domains
# inter_domain_edge_colors = {
#     ('Influencer', 'Study'): 'purple',
#     ('Influencer', 'News'): 'orange',
#     ('Study', 'News'): 'black'
# }

# # Define colors for each domain
# color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# # [Rest of your data processing code]

# # Manually add inter-domain edges with the correct common commenter counts
# inter_domain_edges_counts = {
#     ('Influencer', 'Study'): 1175,
#     ('Influencer', 'News'): 1042,
#     ('Study', 'News'): 570
# }

# for domain_pair, weight in inter_domain_edges_counts.items():
#     edge_color = inter_domain_edge_colors.get(domain_pair, 'black')
#     combined_video_graph.add_edge(domain_pair[0], domain_pair[1], weight=weight, color=edge_color)

# # Arrange domains in a triangular layout
# domain_positions = {
#     'Influencer': (0, 0),
#     'News': (3, 0),  # Increase the distance for better visibility
#     'Study': (1.5, np.sqrt(3))
# }

# # Arrange nodes in a circular layout within big circles
# def arrange_in_circle(graph, domain_positions, domain, radius=1):
#     nodes = [node for node in graph.nodes if isinstance(node, tuple) and node[1] == domain]
#     num_nodes = len(nodes)
#     angle_step = 2 * np.pi / num_nodes
#     positions = {}
#     for i, node in enumerate(sorted(nodes, key=lambda x: x[0])):  # Sort nodes for consistent ordering
#         angle = i * angle_step
#         positions[node] = (np.cos(angle) * radius + domain_positions[domain][0], 
#                            np.sin(angle) * radius + domain_positions[domain][1])
#     return positions

# # Calculate positions for each domain's nodes
# all_positions = {}
# for domain in domain_positions:
#     domain_pos = arrange_in_circle(combined_video_graph, domain_positions, domain)
#     all_positions.update(domain_pos)

# # For domain nodes, use the predefined domain positions
# for node in domain_positions:
#     all_positions[node] = domain_positions[node]

# # Visualization
# plt.figure(figsize=(20, 15))
# edges = combined_video_graph.edges(data=True)

# # Draw nodes with domain-based coloring
# for node in combined_video_graph.nodes():
#     nx.draw_networkx_nodes(combined_video_graph, all_positions, nodelist=[node], 
#                            node_color=color_map[node[1]] if isinstance(node, tuple) else color_map[node], node_size=1000)

# # Draw intra-domain edges
# nx.draw_networkx_edges(combined_video_graph, all_positions, 
#                        edgelist=[(u, v) for u, v, d in edges if u[1] == v[1]], 
#                        width=1)

# # Draw inter-domain edges with distinct colors
# nx.draw_networkx_edges(combined_video_graph, all_positions, 
#                        edgelist=[(u, v) for u, v, d in edges if u[1] != v[1]], 
#                        edge_color='grey', style='dashed', width=2)

# # Edge labels for inter-domain edges
# nx.draw_networkx_edge_labels(combined_video_graph, all_positions, 
#                              edge_labels={(u, v): d['weight'] for u, v, d in edges if u[1] != v[1]}, font_size=8)

# # Domain labels
# for domain, position in domain_positions.items():
#     plt.text(position[0], position[1], domain, fontsize=15, ha='center')

# plt.title("Video Interactions Within and Between Different Domains")
# plt.axis("off")
# plt.show()


# In[32]:


# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt

# # Define file paths for the three Excel files
# file_paths = {
#     'Influencer': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Influencer_Channels_comments.xlsx',
#     'News': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\News_Channels_Comments.xlsx',
#     'Study': r'C:\Users\saura\OneDrive\Desktop\IP_APIs_Work\Study_Channels_comments.xlsx'
# }

# # Load the data from each Excel file into dictionaries of DataFrames
# domain_dataframes = {
#     domain: pd.read_excel(file_path, sheet_name=None) 
#     for domain, file_path in file_paths.items()
# }

# # Create network graphs for each domain
# domain_graphs = {}
# for domain, dfs in domain_dataframes.items():
#     graph = nx.Graph()
#     for sheet_name, df in dfs.items():
#         if 'Video' in sheet_name:  # Filter out irrelevant sheets
#             for other_sheet, other_df in dfs.items():
#                 if 'Video' in other_sheet and sheet_name != other_sheet:
#                     # Find common commenters between sheets
#                     common_commenters = set(df['Channel URL'].dropna()).intersection(set(other_df['Channel URL'].dropna()))
#                     if common_commenters:
#                         # Add an edge for common commenters
#                         graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
#     domain_graphs[domain] = graph

# # Visualization with subplots and edge weights
# fig, axs = plt.subplots(1, 3, figsize=(20, 7))  # 1 row, 3 columns

# # Define colors for each domain
# color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

# for (domain, graph), ax in zip(domain_graphs.items(), axs):
#     # The node size will be 100 times the degree of the node to make it visible
#     node_size = [100 * graph.degree[node] for node in graph]
#     # The edge width will be the weight attribute
#     edge_width = [graph[u][v]['weight'] for u, v in graph.edges()]

#     pos = nx.spring_layout(graph, seed=42)  # Consistent layout for all domains
#     nx.draw_networkx(graph, pos, ax=ax, with_labels=True, node_color=color_map[domain],
#                      node_size=node_size, edge_color='gray', width=edge_width, font_size=10)

#     # Adding edge labels for weights
#     edge_weights = nx.get_edge_attributes(graph, 'weight')
#     nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, font_size=8, ax=ax)

#     ax.set_title(f"{domain} Domain")
#     ax.axis("off")

# plt.suptitle("Common Commenters Across Different Domains")
# plt.tight_layout()
# plt.show()


# In[38]:


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
domain_dataframes = {
    domain: pd.read_excel(file_path, sheet_name=None) 
    for domain, file_path in file_paths.items()
}

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
                    if common_commenters:
                        # Add an edge for common commenters
                        graph.add_edge(sheet_name, other_sheet, weight=len(common_commenters))
    domain_graphs[domain] = graph

# Visualization with subplots for each domain
fig, axs = plt.subplots(1, 3, figsize=(20, 7))  # 1 row, 3 columns

# Define colors for each domain
color_map = {'Influencer': 'lightblue', 'News': 'lightgreen', 'Study': 'lightcoral'}

for (domain, graph), ax in zip(domain_graphs.items(), axs):
    # Calculate node sizes based on the degree of each node
    degrees = dict(graph.degree())
    max_degree = max(degrees.values())
    min_degree = min(degrees.values())
    
    if max_degree == min_degree:  # Avoid division by zero
        size_step = 1  # Set a default size step
    else:
        size_range = max_degree - min_degree
        size_step = 0.1 if size_range > 10 else 1 / size_range  # Adjust size step based on range
    
    node_sizes = {node: 1 - (max_degree - degree) * size_step for node, degree in degrees.items()}
    # Convert cm to points for matplotlib (1cm = 28.45 points)
    node_sizes = {node: (size * 28.45) ** 2 for node, size in node_sizes.items()}  # Area of the node circle

    # Calculate edge widths based on the weight attribute
    weights = nx.get_edge_attributes(graph, 'weight')
    max_weight = max(weights.values()) if weights else 1
    edge_widths = [weights.get(edge, 1) / max_weight * 5 for edge in graph.edges()]

    # Position nodes using the spring layout
    pos = nx.spring_layout(graph, k=0.15, iterations=20)

    # Draw nodes and edges
    nx.draw_networkx_nodes(graph, pos, node_size=[node_sizes[node] for node in graph.nodes()],
                           node_color=color_map[domain], ax=ax)
    nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color='grey', ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax)

    ax.set_title(f"{domain} Domain")
    ax.axis("off")

plt.suptitle("Common Commenters Across Different Domains")
plt.tight_layout()
plt.show()


# In[47]:


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

# Calculate node sizes and edge widths based on the graph's connectivity
degrees = dict(combined_video_graph.degree())
max_degree = max(degrees.values())
node_sizes = {node: (degrees[node] / max_degree) * 1000 + 100 for node in combined_video_graph.nodes()}  # The node size is scaled between 100 and 1100

# Generate positions for the nodes using a layout that respects weights
pos = nx.spring_layout(combined_video_graph, weight='weight', iterations=50)

# Visualization
plt.figure(figsize=(20, 15))

# Draw nodes with domain-based coloring and scaled sizes
nx.draw_networkx_nodes(
    combined_video_graph, pos,
    node_size=[node_sizes[node] for node in combined_video_graph.nodes()],
    node_color=[color_map.get(node[1], 'grey') for node in combined_video_graph.nodes()],
    alpha=0.9
)

# Draw edges with scaled widths
edge_weights = nx.get_edge_attributes(combined_video_graph, 'weight')
for (u, v, data) in combined_video_graph.edges(data=True):
    edge_color = data['color'] if 'color' in data else 'black'
    edge_width = (data['weight'] / max(edge_weights.values())) * 10
    nx.draw_networkx_edges(
        combined_video_graph, pos,
        edgelist=[(u, v)],
        width=edge_width,
        edge_color=edge_color
    )

# Draw labels
nx.draw_networkx_labels(
    combined_video_graph, pos,
    labels={node: node for node in combined_video_graph.nodes()},
    font_size=10
)

plt.title("Common Commenters (Based on Channel URL) Within and Between Videos of Different Domains")
plt.axis('off')
plt.show()


# In[51]:


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

# Calculate node sizes and edge widths based on the graph's connectivity
degrees = dict(combined_video_graph.degree())
max_degree = max(degrees.values())
# node_sizes = {node: (degrees[node] / max_degree) * 1000 + 100 for node in combined_video_graph.nodes()}  # The node size is scaled between 100 and 1100
node_sizes = {
    node: 100 + 1900 * ((node_weights[node] - min_weight) / (max_weight - min_weight)) 
    for node in combined_video_graph.nodes()
}
# Generate positions for the nodes using a layout that respects weights
pos = nx.spring_layout(combined_video_graph, weight='weight', iterations=50)

# Visualization
plt.figure(figsize=(20, 15))

# Draw nodes with domain-based coloring and updated sizes
nx.draw_networkx_nodes(
    combined_video_graph, pos,
    node_size=[node_sizes[node] for node in combined_video_graph.nodes()],
    node_color=[color_map.get(node[1], 'grey') for node in combined_video_graph.nodes()],
    alpha=0.9
)

# Draw edges with scaled widths
for (u, v, data) in combined_video_graph.edges(data=True):
    edge_color = data['color'] if 'color' in data else 'black'
    edge_width = (data['weight'] / max(edge_weights.values())) * 10
    nx.draw_networkx_edges(
        combined_video_graph, pos,
        edgelist=[(u, v)],
        width=edge_width,
        edge_color=edge_color
    )

plt.title("Common Commenters (Based on Channel URL) Within and Between Videos of Different Domains")
plt.axis('off')
plt.show()


# In[52]:


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

# Calculate node sizes and edge widths based on the graph's connectivity
degrees = dict(combined_video_graph.degree())
max_degree = max(degrees.values())
# node_sizes = {node: (degrees[node] / max_degree) * 1000 + 100 for node in combined_video_graph.nodes()}  # The node size is scaled between 100 and 1100
node_sizes = {
    node: 100 + 1900 * ((node_weights[node] - min_weight) / (max_weight - min_weight)) 
    for node in combined_video_graph.nodes()
}
# Generate positions for the nodes using a layout that respects weights
pos = nx.spring_layout(combined_video_graph, weight='weight', iterations=50)

# Visualization
plt.figure(figsize=(20, 15))

# Draw nodes with domain-based coloring and updated sizes
nx.draw_networkx_nodes(
    combined_video_graph, pos,
    node_size=[node_sizes[node] for node in combined_video_graph.nodes()],
    node_color=[color_map.get(node[1], 'grey') for node in combined_video_graph.nodes()],
    alpha=0.9
)

# Draw edges with scaled widths
for (u, v, data) in combined_video_graph.edges(data=True):
    edge_color = data['color'] if 'color' in data else 'black'
    edge_width = (data['weight'] / max(edge_weights.values())) * 10
    nx.draw_networkx_edges(
        combined_video_graph, pos,
        edgelist=[(u, v)],
        width=edge_width,
        edge_color=edge_color
    )

plt.title("Common Commenters (Based on Channel URL) Within and Between Videos of Different Domains")
plt.axis('off')
plt.show()


# In[16]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community  # Add this import for modularity calculation

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
                        graph.add_edge((sheet_name, domain), (other_sheet, domain), weight=len(common_commenters))
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
        combined_video_graph.add_edge(u, v, weight=data['weight'], color=color_map[domain])

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

# Calculate modularity-based communities using the Louvain method
communities = community.best_partition(combined_video_graph)
nx.set_node_attributes(combined_video_graph, communities, 'community')

# Calculate node sizes based on the graph's connectivity
centrality = nx.degree_centrality(combined_video_graph)
max_centrality = max(centrality.values())
node_sizes = {node: (centrality[node] / max_centrality) * 1000 + 100 for node in combined_video_graph.nodes()}  # The node size is scaled between 100 and 1100

# Generate positions for the nodes using a layout that respects weights
pos = nx.spring_layout(combined_video_graph, weight='weight', iterations=50)

# Visualization
plt.figure(figsize=(20, 15))

# Draw nodes with community-based coloring and updated sizes
modularity_colors = {community: plt.cm.tab10(community) for community in set(nx.get_node_attributes(combined_video_graph, 'community').values())}
nx.draw_networkx_nodes(
    combined_video_graph, pos,
    node_size=[node_sizes[node] for node in combined_video_graph.nodes()],
    node_color=[modularity_colors[nx.get_node_attributes(combined_video_graph, 'community')[node]] for node in combined_video_graph.nodes()],
    alpha=0.9
)

# Draw edges with scaled widths
edge_weights = nx.get_edge_attributes(combined_video_graph, 'weight')
for (u, v, data) in combined_video_graph.edges(data=True):
    edge_color = data['color'] if 'color' in data else 'black'
    edge_width = (data['weight'] / max(edge_weights.values())) * 10
    nx.draw_networkx_edges(
        combined_video_graph, pos,
        edgelist=[(u, v)],
        width=edge_width,
        edge_color=edge_color
    )

# Annotate nodes with their community labels for better identification
node_labels = nx.get_node_attributes(combined_video_graph, 'community')
nx.draw_networkx_labels(
    combined_video_graph, pos,
    labels={node: f'{node[0]}\n({node_labels[node]})' for node in combined_video_graph.nodes()},
    font_size=10
)

# Highlight News nodes in yellow
news_nodes = [node for node in combined_video_graph.nodes() if node[1] == 'News']
nx.draw_networkx_nodes(
    combined_video_graph, pos,
    nodelist=news_nodes,
    node_size=[node_sizes[node] for node in news_nodes],
    node_color='yellow',
    alpha=0.9
)

plt.title("Common Commenters (Based on Channel URL) Within and Between Videos of Different Domains with Modularity")
plt.axis('off')
plt.show()


# In[18]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community  # Ensure you have the 'python-louvain' package installed

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

# Create a combined graph for all videos from all domains
combined_video_graph = nx.Graph()

# Add intra-domain edges (within the same domain)
for domain, graph in domain_graphs_url.items():
    for u, v, data in graph.edges(data=True):
        combined_video_graph.add_edge((u, domain), (v, domain), weight=data['weight'])

# Add inter-domain edges (between videos of different domains)
for video1, commenters1 in video_commenters.items():
    for video2, commenters2 in video_commenters.items():
        if video1[1] != video2[1]:  # Only for different domains
            common_commenters = commenters1.intersection(commenters2)
            if len(common_commenters) > 0:
                combined_video_graph.add_edge(video1, video2, weight=len(common_commenters))

# Use the Louvain community detection algorithm to find clusters
partition = community.best_partition(combined_video_graph, weight='weight')

# Calculate node sizes based on the number of common users
node_sizes = {
    node: 100 + 1900 * (len(commenters) / len(video_commenters[node])) 
    for node, commenters in video_commenters.items()
}

# Generate positions for the nodes using a layout that respects weights
pos = nx.spring_layout(combined_video_graph, weight='weight', iterations=50)

# Visualization
plt.figure(figsize=(20, 15))

# Draw nodes with community-based coloring and scaled sizes
node_colors = [partition[node] for node in combined_video_graph.nodes()]
nx.draw_networkx_nodes(
    combined_video_graph, pos,
    node_size=[node_sizes[node] for node in combined_video_graph.nodes()],
    node_color=node_colors,
    cmap=plt.cm.get_cmap('viridis'),
    alpha=0.9
)

edge_weights = nx.get_edge_attributes(combined_video_graph, 'weight')
for (u, v, data) in combined_video_graph.edges(data=True):
    edge_width = (data['weight'] / max(edge_weights.values())) * 10
    nx.draw_networkx_edges(
        combined_video_graph, pos,
        edgelist=[(u, v)],
        width=edge_width,
        edge_color='grey',
        alpha=0.2
    )

# Draw labels
nx.draw_networkx_labels(
    combined_video_graph, pos,
    labels={node: node[0] for node in combined_video_graph.nodes()},
    font_size=8
)

plt.title("Common Commenters Clustering and Size based on Channel URL")
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:





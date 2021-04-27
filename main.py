#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import geopandas as gpd
import folium


#In[2]:


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# In[4]:


import io
import pybase64 as base64

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = 'Black'
plt.rcParams['axes.labelcolor'] = '#909090'
plt.rcParams['xtick.color'] = 'Black'
plt.rcParams['ytick.color'] = 'Black'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.facecolor'] = 'none'
# # DASH
# Load Data
state_df = pd.read_csv('State_centroids.csv')
india_clean = gpd.read_file('INDIA_SIMPLIFIED/INDIA_SIMPLIFIED.shp')
mega = pd.read_csv('MEGA.csv')
mega['State_name'] = mega['State_name'].str.strip()
mega['District_name'] = mega['District_name'].str.strip()
india_clean['State'] = india_clean['State'].str.strip()
india_clean['District'] = india_clean['District'].str.strip()
state_df['State_name'] = state_df['State_name'].str.strip()


# Build App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
server = app.server
app.title = "How deprived are Indian districts? - Mission Antyodaya 2019"
app.layout = html.Div([
    html.Div([
        html.H1("Mission Antyodaya 2019"),
        html.Label('Select Type of Metric', style={'color': '#FFFFFF', 'fontSize': '16px'}),
        dcc.Dropdown(id='Type of Metric',
                     options=[
                         {"label": "Clusters", "value": "labels"},
                         {"label": "Deprivation Index", "value": "Deprivation Index"}],
                     placeholder='Choose a metric',
                     multi=False,
                     value="Deprivation Index",
                     style={'width': "50%", 'verticalAlign': "middle", 'color':'black'}
                     ),
    
    
    html.Label('Select State', style={'color' : '#FFFFFF', 'fontSize': '16px'}),
    dcc.Dropdown(id='State',
                options=[{'label': i, 'value': i} for i in state_df.State_name.unique()],
                placeholder= 'Select State',
                multi=False,
                searchable=True,
                value ="ANDHRA PRADESH",
                clearable=True,
                style={'width': "50%",'verticalAlign':"middle",'color':'black'}),
                     
                        ],
style=dict(display='flex')
),

    html.Div([html.P("Performance of Rural India in Education, Health, Banking and Infrastructure"),
    
    html.Label('Select Index', style={'color' : '#FFFFFF', 'fontSize': '16px'}),
    dcc.Dropdown(id='Type of Index',
                options=[
                    {"label":"Infrastructure","value":"Infrastructure"},
                    {"label":"Education","value":"Education"},
                    {"label":"Health","value":"Health"},
                    {"label":"Financial","value":"Banking"}],
                placeholder= 'Choose Index',
                multi=False,
                value="Infrastructure",
                clearable=True,
                style={'width': "50%",'verticalAlign':"middle",'color':'black'},
                ),
     
    html.Label('Choose Deprivation Threshold',style={'color' : '#FFFFFF', 'fontSize': '16px'}),
        dcc.Dropdown(id='DeprivationThreshold',
                options=[{'label':0.1,"value":0.1},
                         {'label':0.3,"value":0.3},
                         {'label':0.5,"value":0.5}],
                placeholder= 'Select Deprivation Threshold',
                multi=False,
                searchable=False,
                value = 0.3,
                clearable=True,
                style={'width': "50%",'verticalAlign':"middle",'color':'black'}),
    
    ],
style=dict(display='flex')
),

html.Div(
[html.Img(id="pie",width='45%',height='500'),
 html.Iframe(id='map', srcDoc=None,width='45%',height='500'),
 #html.Img(id="pie",width='45%',height='250')
],
style=dict(display='flex')
)

]
) 



# Define callback to update map
@app.callback(
    [Output("pie","src"),
     Output('map', 'srcDoc')
    ],
    [Input("Type of Metric", "value"),
     Input("State", "value"),
     Input("Type of Index","value"),
     Input("DeprivationThreshold","value")]
)

########################################
def update_figure(mtric_chosen,state,index,deprivation_threshold):

    if index==None:
        percentages = pd.read_csv('COMPOSITE.csv')

    else:
        percentages = pd.read_csv(str(index)+'.csv')
        percentages['State_name'] = percentages['State_name'].str.strip()
        percentages['District_name'] = percentages['District_name'].str.strip()
        percentages['MAP_KEY'] = percentages['MAP_KEY'].str.strip()

    demog_cols = ['Year','State_name', 'District_name','State_code', 'District_code','villages_surveyed', 'tot_pop', 'pop_male', 'pop_female', 'tot_hh','MAP_KEY']
    dep_matrix = percentages[list(set(percentages.columns) - set(demog_cols))]
    dep_matrix = (dep_matrix<deprivation_threshold).astype(int)
    dep_matrix['Deprivation Index'] = dep_matrix.mean(axis=1)
    dep_df = pd.concat([percentages[demog_cols],dep_matrix],axis=1)

    demog_cols.append('Deprivation Index')
    X_dep = dep_df.drop(demog_cols,axis=1).values

    pca = PCA().fit(X_dep)
    variances_explained = np.cumsum(pca.explained_variance_ratio_).tolist()
    for idx,var in enumerate(variances_explained):
        if var>0.9:
            n_components = idx+1
            break
    pca_dep = PCA(n_components = n_components).fit(X_dep).transform(X_dep)

    kmeans = KMeans(n_clusters=3, random_state=1).fit(pca_dep)

    #Attach cluster to dataset
    labelled_KM_PCA = pd.concat((dep_df,pd.DataFrame(kmeans.labels_)),axis=1).rename({0:'labels'},axis=1)


    SUBINDEX_DF = pd.merge(percentages,labelled_KM_PCA[['District_code','labels','Deprivation Index']],on='District_code').sort_values(by='Deprivation Index')
    kMeans_df =  pd.merge(india_clean,SUBINDEX_DF,left_on=['State','District'],right_on=['State_name','MAP_KEY'],how='right')
    kMeans_df = kMeans_df[['State_name','District_name','District_code','labels','Deprivation Index','geometry']].dropna()

    #WORST
    worst_label = SUBINDEX_DF['labels'][-1:].values[0]
    #BEST
    best_label = SUBINDEX_DF['labels'][:1].values[0]
    #MIDDLE
    middle_label = list(set(SUBINDEX_DF['labels'].unique().tolist())-set([worst_label,best_label]))[0]

    kMeans_df.loc[(kMeans_df.labels == worst_label),"labels"]= "worst"
    kMeans_df.loc[(kMeans_df.labels == best_label),"labels"]= "best"
    kMeans_df.loc[(kMeans_df.labels == middle_label),"labels"]= 'medium'
    kMeans_df.loc[(kMeans_df.labels == "worst"),"labels"]= 2
    kMeans_df.loc[(kMeans_df.labels == "best"),"labels"]= 0
    kMeans_df.loc[(kMeans_df.labels == "medium"),"labels"]= 1

    kMeans_df['Deprivation Index'] = kMeans_df['Deprivation Index'].round(decimals=2)
    ### folium
    if state==None:
        center = [22.89259,79.54238]
        map_data = kMeans_df.copy()
        zoom_start=4
        Mega1 = mega
        state_dep = pd.DataFrame(dep_df.drop(demog_cols, axis=1).mean(), columns=['Deprivation'])
    else:
        center=state_df[state_df.State_name ==state][['lat','lon']].values.tolist()[0]
        map_data = kMeans_df[kMeans_df.State_name==state].copy()
        zoom_start=6
        Mega1 = mega[mega['State_name'] == state]
        state_dep = pd.DataFrame(dep_df[dep_df['State_name'] == state].drop(demog_cols, axis=1).mean(),
                                 columns=['Deprivation'])

    m = folium.Map(location=center,zoom_start=zoom_start,tiles="OpenStreetMap")
    choropleth = folium.Choropleth(
        geo_data=map_data,
        name= 'MAP',
        data = map_data,
        columns=['District_code', str(mtric_chosen)],
        key_on='feature.properties.District_code',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=1,
        legend_name=str(mtric_chosen)).add_to(m)


    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=['State_name','District_name', "Deprivation Index","labels"],
            aliases=['State:','District:',"Deprivation Index","Cluster"],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")))



    m.save('INFRA1.html')





    #plots
    state_dep = state_dep/state_dep[state_dep.columns].sum()*100
    state_dep = state_dep[state_dep['Deprivation']!=0].sort_values(by='Deprivation',ascending=True).reset_index()
    state_dep = state_dep[-10:]

    indices = ['Banking','Education','Health','Infrastructure']

    buf = io.BytesIO()
    fig, (ax1,ax2) = plt.subplots(2,figsize=(8,5))
    ax1.pie(Mega1[['Banking','Education','Health','Infrastructure']].mean(axis=0).values.tolist(),labels=indices,autopct='%1.0f%%',shadow=False, startangle=0)
    ax1.axis('equal')


    state_dep.plot(x='index',y='Deprivation',kind='barh',ax=ax2,color="maroon")
    ax2.set_ylabel('Top 10 Factors',fontsize=10)
    ax2.set_xlabel('Contribution to Deprivation (%)',fontsize=10)
    ax2.get_legend().remove()


    plt.suptitle("Dominance of factors in Deprivation of the State",fontweight='bold',fontsize=20)


    plt.tight_layout()
    plt.savefig(buf,format = 'png')
    plt.close()


    data = base64.b64encode(buf.getbuffer()).decode("utf8")


    return "data:image/png;base64,{}".format(data),open('INFRA1.html', 'r').read()
# Run app and display result inline in the notebook

if __name__ == "__main__":
    app.run_server(debug=False,use_reloader=False)



# In[ ]:





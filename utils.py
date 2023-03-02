"""
UTILS 
- Helper functions to use for your recommender funcions, etc
- Data: import files/models here e.g.
    - movies: list of movie titles and assigned cluster
    - ratings
    - user_item_matrix
    - item-item matrix 
- Models:
    - nmf_model: trained sklearn NMF model
"""
import pandas as pd
import numpy as np
import pickle 

# load data
movies = pd.read_csv('data/movies_new.csv', sep=';')
ratings = pd.read_csv('data/ratings_new.csv')

# load models
with open('models/nmf_recommender.pkl', 'rb') as file:
    model_nmf = pickle.load(file)

with open('models/distance_recommender.pkl', 'rb') as file:
    model_cos = pickle.load(file)

# load sparse user-item matrix R
with open('data/R.pkl', 'rb') as file:
    R = pickle.load(file)

example_query = {
    # movieId, rating
    3:5, 
    18:5,
    194:5,
    276:5,
    401:5,
    595:5,
    616:5,
    1200:5
}



def movie_to_id(string_titles):
    '''
    converts movie title to id for use in algorithms'''
    
    movieID = movies.set_index('title').loc[string_titles]['movieId']
    movieID = movieID.tolist()
    
    return movieID

def id_to_movie(movieID):
    '''
    converts movie Id to title
    '''
    rec_title = movies.set_index('movieId').loc[movieID]['title']
    
    return rec_title


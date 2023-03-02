"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
from utils import movies
from utils import model_nmf
from utils import model_cos
from utils import ratings
from utils import R
from scipy.sparse import csr_matrix
from sklearn import neighbors
from utils import example_query
from utils import id_to_movie

def recommend_random(k=3):
    return movies['title'].sample(k).to_list()

def recommend_with_NMF(query, model = model_nmf, k=3):
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model
    OUTPUT
    - a list of movieIds
    """
    # 1. candiate generation
    # construct a user vector
    data=list(query.values())      # the ratings of the new user
    row_ind=[0]*len(data)          # we use just a single row 0 for this user
    col_ind=list(query.keys())  
    data, row_ind,col_ind
    # new user vector: needs to have the same format as the training data
    user_vec=csr_matrix((data, (row_ind, col_ind)), shape=(1, R.shape[1]))
    
    # 2. scoring
    # calculate the score with the NMF model
    scores=model.inverse_transform(model.transform(user_vec))
    scores=pd.Series(scores[0]) # convert to pandas series
    
    # 3. ranking
    scores[query.keys()]=0 # give a zero score to movies the user has allready seen
    scores=scores.sort_values(ascending=False) # sort the scores from high to low 
   
    # return the top-k highst rated movie ids or titles
    recommendations=scores.head(k).index
    rec_titles = []
    for movieID in recommendations:
        title = id_to_movie(movieID)
        rec_titles.append(title)

    return rec_titles

    
def recommend_neighborhood_old(query, model=model_cos, k=3):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """   
        # new user vector: needs to have the same format as the training data
    # pre fill it with zeros
    user_vec = np.repeat(0, R.shape[1])

    # fill in the ratings that arrived from the query
    user_vec[list(query.keys())] = list(query.values())

    
    # calculates the distances to all other users in the data!
    similarity_scores, neighbor_ids = model.kneighbors(
    [user_vec],
    n_neighbors=10,
    return_distance=True
    )

    # sklearn returns a list of predictions
    # extract the first and only value of the list

    neighbors = pd.DataFrame(
    data = {'neighbor_id': neighbor_ids[0], 'similarity_score': similarity_scores[0]}
    )

    neighbors.sort_values(
    by='similarity_score',
    ascending=False,
    inplace=True,
    ignore_index=True
    )
    
    # only look at ratings for users that are similar!
    neighborhood = ratings[ratings['userId'].isin(neighbors['neighbor_id'])]
    
    # calculate the summed up rating for each movie
    # summing up introduces a bias for popular movies
    # averaging introduces bias for movies only seen by few users in the neighboorhood
    df_score = neighborhood.groupby('movieId')[['rating']].sum()
    df_score.rename(columns={'rating': 'score'}, inplace=True)
    df_score.reset_index(inplace=True)
    
    # give a zero score to movies the user has allready seen
    #df_score['score'] = df_score['score'].map(lambda x: 0 if x in query.keys() else x)
    df_score['score'] = df_score['score'].map({query.keys:0})


    # sort the scores from high to low 
    df_score.sort_values(
    by='score',
    ascending=False,
    inplace=True,
    ignore_index=True
    )
    # get top k scores
    top_scores = df_score.head(k)
    
    top_movies = movies[movies['movieId'].isin(top_scores['movieId'])]
    return top_movies['title'].to_list()
    
def recommend_neighborhood_(query, model= model_cos, ratings=ratings, n=10, k=5):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """
    # 1. candiate generation
    # construct a user vector
    user_vec = np.repeat(0, R.shape[1])

    # fill in the ratings that arrived from the query
    user_vec[list(query.keys())] = list(query.values())
   
    # 2. scoring
    # find n neighbors
    userIds = model.kneighbors([user_vec], n_neighbors=n, return_distance=False)[0]
    scores = ratings.set_index('userId').loc[userIds].groupby('movieId')['rating'].sum()
    
    # 3. ranking
    # filter out movies allready seen by the user
    scores[query.keys()]=0 
    scores=scores.sort_values(ascending=False)
    
     # return the top-k highst rated movie ids or titles
    recommendations=scores.head(k).index
    rec_titles = []
    for movieID in recommendations:
        title = id_to_movie(movieID)
        rec_titles.append(title)

    return rec_titles


def recommend_neighborhood(query, model=model_cos, ratings=ratings, n = 10, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """
    # 1. candiate generation
    # construct a user vector
    # new user vector: needs to have the same format as the training data
    # pre fill it with zeros
    user_vec = np.repeat(0, R.shape[1])

    # fill in the ratings that arrived from the query
    user_vec[list(query.keys())] = list(query.values())
    
   
    # 2. scoring
    # calculates the distances to all other users in the data!
    distances, userIds = model.kneighbors([user_vec], n_neighbors=n, return_distance=True)

    # sklearn returns a list of predictions - extract the first and only value of the list
    distances = distances[0]
    userIds = userIds[0]
    # only look at ratings for users that are similar!
    neighborhood = ratings.set_index('userId').loc[userIds]
    
    
    # 3. ranking
    scores =  neighborhood.loc[userIds].groupby('movieId')['rating'].sum()

    # filter out movies allready seen by the user
    scores[query.keys()]=0 
    scores=scores.sort_values(ascending=False)

    # return the top-k highst rated movie ids or titles
    recommendations=scores.head(k).index
    rec_titles = []
    for movieID in recommendations:
        title = id_to_movie(movieID)
        rec_titles.append(title)

    return rec_titles
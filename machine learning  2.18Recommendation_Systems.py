# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 00:14:11 2022

@author: HAMZA
"""

import pandas as pd


# import movie data set and look at columns
movie = pd.read_csv("movie.csv")
movie.columns

# what we need is that movie id and title
movie = movie.loc[:,["movieId","title"]]
movie.head(10)

#%%

rating = pd.read_csv("rating.csv")
rating.columns

# what we need is that user id, movie id and rating
rating = rating.loc[:,["userId","movieId","rating"]]
rating.head(10)

#%%

# then merge movie and rating data
data = pd.merge(movie,rating)

data.head(10)


data.shape
data = data.iloc[:1000000,:]

# lets make a pivot table in order to make rows are users and columns are movies. And values are rating
pivot_table = data.pivot_table(index = ["userId"],columns = ["title"],values = "rating")
pivot_table.head(10)


pivot_table = data.pivot_table(index = ["userId"],columns = ["title"],values = "rating")
pivot_table.head(10)

movie_watched = pivot_table["Bad Boys (1995)"]
similarity_with_other_movies = pivot_table.corrwith(movie_watched)  # find correlation between "Bad Boys (1995)" and other movies
similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending=False)
similarity_with_other_movies.head()

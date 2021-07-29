from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import sparse
import recommendation_system


def update_similar_video():
    try:
        recommendation_system.get_similarity_matrix()
        print('Similarity matrix was updated')
    except:
        print('ERROR: Similarity matrix was not updated')

def get_similar_book(book_id):
    try:
        book_id = book_id
        titles = pd.read_csv('titles.csv',index_col=0)
        similar_book_dict = np.load('similar_book_dict.npy',allow_pickle='TRUE').item()
        print("Most familiar for book",titles.loc[book_id])
        print(titles.loc[similar_book_dict[book_id][:10]])
    except:
        print('ERROR: Unable to get similar video')


def update_similar_user_book():
    try:
        recommendation_system.get_user_user_recommendation()
        print('Recommendations was updated')
    except:
        print('ERROR: Recommendations was not updated')


def get_similar_user_book():
    try:
        top_10_popular = pd.read_csv('top_10_popular.csv')
        print(top_10_popular)
    except:
        print('ERROR: Unable to get similar video by user')


def update_all():
    try:
        recommendation_system.get_user_user_recommendation()
        recommendation_system.get_similarity_matrix()
        print('Everything was updated')
    except:
        print('ERROR: Everything was not updated')


update_similar_user_book()
update_similar_video()
get_similar_book(1350)
get_similar_user_book()


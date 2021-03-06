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
        return(titles.loc[similar_book_dict[book_id][:10]])
    except:
        print('ERROR: Unable to get similar book')
        print('Returning top 10 vids')
        get_top_10_popular()


def update_similar_user_book():
    try:
        recommendation_system.get_user_user_recommendation()
        print('Recommendations was updated')
    except:
        print('ERROR: Recommendations was not updated')


def get_similar_user_book():
    try:
        total_preds = pd.read_csv('total_preds.csv')
        return(total_preds)
    except:
        print('ERROR: Unable to get similar video by user')


def get_top_10_popular():
    try:
        top_10_popular = pd.read_csv('top_10_popular.csv')
        return(top_10_popular['top_10_vids'].values)
    except:
        print('ERROR: Unable to get top 10 popular')


def update_all():
    try:
        recommendation_system.get_user_user_recommendation()
        recommendation_system.get_similarity_matrix()
        print('Everything was updated')
    except:
        print('ERROR: Everything was not updated')


total_preds = pd.read_csv('total_preds.csv')
similar_book_dict = np.load('similar_book_dict.npy',allow_pickle='TRUE').item()
titles = pd.read_csv('titles.csv')
final_system = {}
for index, row in total_preds.iterrows():
        for i in row.values[1:]:
            l=[]
            try:
                l.extend(titles.loc[similar_book_dict[(titles.loc[titles['id'] == i]).index[0]][:10]]['id'].values)
            except:
                top_10_popular = pd.read_csv('top_10_popular.csv')
                l.extend(list(top_10_popular['top_10_vids'].values))
        final_system[index] = set(l)
print(final_system)



from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def get_similarity_matrix():
    """Book-Book recommendations"""

    items = pd.read_csv('Data/items.csv')
    s = pd.get_dummies(items.genres.str.split(',', expand=True))
    items = pd.concat([items, s], axis=1)
    items.drop(["genres"], inplace=True, axis=1)
    items.drop(["authors"], inplace=True, axis=1)
    titles = items["title"]
    items.drop(["title"], inplace=True, axis=1)
    items.drop(["year"], inplace=True, axis=1)
    items.set_index(items.columns[0], inplace=True)
    titles.to_csv('titles.csv')

    TestUISparseData = sparse.csr_matrix(items.values)
    m_m_similarity = cosine_similarity(TestUISparseData.T, dense_output=False)

    book_ids = np.unique(m_m_similarity.nonzero())

    similar_book_dict = dict()
    for book in book_ids:
        smlr = np.argsort(-m_m_similarity[book].toarray().ravel())[1:10]
        similar_book_dict[book] = smlr
    np.save('similar_book_dict.npy', similar_book_dict)


def get_user_user_recommendation():
    import pandas as pd
    import numpy as np
    from implicit.nearest_neighbours import BM25Recommender
    import scipy.sparse as sp
    """User-User recommendation"""

    interactions = pd.read_csv('Data/interactions.csv')
    items = pd.read_csv('Data/items.csv')
    users = pd.read_csv('Data/users.csv')

    """Получить топ популярных видео"""

    class Popular_Recommender():
        def __init__(self, max_K=100, days=30, item_column='item_id', dt_column='date'):
            self.max_K = max_K
            self.days = days
            self.item_column = item_column
            self.dt_column = dt_column
            self.recommendations = []

        def fit(self, df):
            min_date = pd.to_datetime(df[self.dt_column]).max().normalize() - pd.DateOffset(days=self.days)
            self.recommendations = df.loc[
                pd.to_datetime(df[self.dt_column]) > min_date, self.item_column].value_counts().head(
                self.max_K).index.values

    def get_popular(interactions, num=10, days=10, dt_column='start_date'):
        pop_rec = Popular_Recommender(days=days, dt_column=dt_column)
        pop_rec.fit(interactions)
        populars = list(pop_rec.recommendations[:num])
        return populars

    populars = get_popular(interactions, 10)
    df = pd.DataFrame(data={"col1": populars})
    df.to_csv("top_10_popular.csv", sep=',', index=False)

    dropped_users = []
    n_interactions = 4
    for user in users['user_id']:
        if users[users['user_id'] == user]['num_its'].item() < n_interactions:
            dropped_users.append(user)


    def get_coo_matrix(interactions,
                       users_mapping,
                       items_mapping,
                       user_col='user_id',
                       item_col='item_id',
                       weight_col=None):
        if weight_col is None:
            weights = np.ones(len(interactions), dtype=np.float32)
        else:
            weights = interactions[weight_col].astype(np.float32)

        interaction_matrix = sp.coo_matrix((
            weights,
            (
                interactions[user_col].map(users_mapping.get),
                interactions[item_col].map(items_mapping.get)
            )
        ))
        return interaction_matrix

    def make_mapping(data):
        return dict([(v, k) for k, v in enumerate(data)])

    items_mapping = make_mapping(items['id'].unique())
    users_mapping = make_mapping(interactions['user_id'].unique())

    items_inv_mapping = dict({(v, k) for k, v in items_mapping.items()})

    interactions_matrix = get_coo_matrix(interactions,
                                         users_mapping=users_mapping,
                                         items_mapping=items_mapping).tocsr()

    imp_model = BM25Recommender(K=10)
    imp_model.fit(interactions_matrix.T)

    top_n = 10

    total_preds = {}
    for user in users['user_id']:
        preds = imp_model.recommend(users_mapping[user], interactions_matrix,
                                    N=top_n, filter_already_liked_items=True)
        preds = [items_inv_mapping[pred[0]] for pred in preds]

        total_preds[user] = preds

    for user in dropped_users:
        total_preds[user] = populars

    total_preds = pd.DataFrame.from_dict(total_preds, orient='index')
    total_preds.to_csv('total_preds.csv')

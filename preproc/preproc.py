import pandas as pd
from collections import defaultdict
import numpy as np
import pickle
import argparse
import os
import sys
import random
import pathlib


def parse_arguments():
    # Command line flags to determine which dataset to pre-process
    parser = argparse.ArgumentParser()
    parser.add_argument('--movielens', dest='process_movielens', action='store_true', default=True,
                        help="Pre-process MovieLens data")
    parser.add_argument('--goodreads', dest='process_goodreads', action='store_true', default=True,
                        help="Pre-process Goodreads data")
    parser.add_argument('--train_test_ratio', dest='train_test_ratio', type=float, default=0.5,
                        help="Train test split ratio")

    return parser.parse_args()


def movielens_preproc():
    '''
    Add metadata and store it as a dataframe
    '''
    # Add headers
    ratings_cols = 'UserID::MovieID::Rating::Timestamp'.split('::')
    movies_cols = 'MovieID::Title::Genres'.split('::')
    users_cols = 'UserID::Gender::Age::Occupation::Zip-code'.split('::')
    # Read data
    ratings = pd.read_csv('../data/ml-1m/ratings.dat', sep='::', engine='python', names=ratings_cols)
    movies = pd.read_csv('../data/ml-1m/movies.dat', sep='::', engine='python', names=movies_cols)
    users = pd.read_csv('../data/ml-1m/users.dat', sep='::', engine='python', names=users_cols)

    # Randomly assign one genre out of multiple to each movie
    gs = movies['Genres'].tolist()
    new_genre_list = []
    for i in range(len(movies)):
        t = gs[i].split('|')
        random.shuffle(t)
        new_genre_list += [t[0]]

    movies = movies.assign(Genre_Assigned=new_genre_list)

    # Merge dataframes
    temp = pd.merge(ratings, users, how='left', on='UserID')
    data = pd.merge(temp, movies, how='left', on='MovieID')

    # Each meta-user is identified by the age group and occupation
    # Below categories as taken from: http://files.grouplens.org/datasets/movielens/ml-1m-README.txt
    ages = [1, 18, 25, 35, 45, 50, 56]

    occupations = list(range(21))

    meta_users = [(x, y) for x in ages for y in occupations]

    # Adding columns for meta-user and genre-ids for easier lookup
    genre_order = ['Mystery', 'Drama', 'Sci-Fi', "Children's", 'Horror', 'Film-Noir', 'Crime', 'Romance',
                   'Fantasy', 'Musical', 'Animation', 'Adventure', 'Action', 'Comedy', 'Documentary', 'War',
                   'Thriller', 'Western']
    # IDs for genres
    genre_ids = dict(zip(genre_order, list(range(18))))

    age = data['Age'].tolist()
    occ = data['Occupation'].tolist()
    genr = data['Genre_Assigned'].tolist()

    # IDs for meta-users
    meta_user_ids = dict(zip(meta_users, list(range(147))))

    meta_user_col = []
    genre_id_col = []
    for i in range(len(data)):
        meta_user_col += [meta_user_ids[(age[i], occ[i])]]
        genre_id_col += [genre_ids[genr[i]]]

    data = data.assign(Meta_User_Col=meta_user_col)
    data = data.assign(genre_col=genre_id_col)

    pathlib.Path('genres').mkdir(parents=False, exist_ok=True)
    data.to_pickle('genres/data_with_id')

    print(f"Pre-processing complete. Processed data dumped at {os.getcwd() + '/genres/data_with_id'}")


def genre_train_test_split(train_test_ratio=0.5):
    genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
              'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
              'War', 'Western']

    data = pd.read_pickle('genres/data_with_id')

    # Get user counts and sort them in reverse
    user_counts = data['UserID'].value_counts().to_dict()
    reverse_sorted_users = sorted([(user_counts[user], user) for user in user_counts], reverse=True)

    # Retain top % as train data and the rest for testing (to simulate real surveys)
    cnt = 0
    users_for_train = []
    for i in range(len(reverse_sorted_users)):
        cnt += reverse_sorted_users[i][0]
        users_for_train.append(reverse_sorted_users[i][1])
        if cnt > train_test_ratio * len(data):
            break

    train_data = data[data['UserID'].isin(users_for_train)]
    all_users = set(data['UserID'])
    users_for_test = all_users.difference(users_for_train)
    test_data = data[data['UserID'].isin(users_for_test)]

    pickle.dump(train_data, open('genres/train_data_usercount', 'wb'))
    pickle.dump(test_data, open('genres/test_data_usercount', 'wb'))

    print(f"Train-test split complete with a train-test ratio of: {train_test_ratio}")


def genre_data_process():
    genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
              'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
              'War', 'Western']

    train_data_id = pd.read_pickle('genres/train_data_usercount')
    test_data_id = pd.read_pickle('genres/test_data_usercount')

    # Constructing conditional expectation tables
    genre_tables = defaultdict(np.ndarray)

    for table_genre in range(0, 18):
        genre_tables[table_genre] = -np.ones((5, len(genres)))  ##Hard-coded rating size

        for rating in range(1, 6):
            users = set(train_data_id[(train_data_id['genre_col'] == table_genre) & \
                                      (train_data_id['Rating'] == rating)]['UserID'])

            genre_tables[table_genre][:, table_genre] = np.arange(1, 6)  # Initialize the table reference column

            for genre in np.delete(np.arange(0, 18), table_genre):
                d = train_data_id[(train_data_id['genre_col'] == genre) & \
                                  (train_data_id['UserID'].isin(users))]

                genre_tables[table_genre][rating - 1, genre] = np.average(d['Rating'])  # Note: Use 'rating-1' to index

    pickle.dump(genre_tables, open('genres/genre_tables_train.pkl', 'wb'))

    # Test data true means
    true_means = np.zeros(18)
    for genre in range(18):
        d = test_data_id[test_data_id['genre_col'] == genre]
        true_means[genre] = np.mean(d['Rating'])

    pickle.dump(true_means, open('genres/true_means_test', 'wb'))


# Prepare dataset to recommend best movie
def movie_train_test_split(train_test_ratio=0.5):
    data = pd.read_pickle('genres/data_with_id')

    # pick top 50 movies with most data
    movie_counts = data['MovieID'].value_counts().to_dict()
    reverse_sorted_movies = sorted([(movie_counts[movie], movie) for movie in movie_counts], reverse=True)
    movies_as_arms = [movie_id for (m_count, movie_id) in reverse_sorted_movies[:50]]
    chosen_data = data[data['MovieID'].isin(movies_as_arms)]

    # assigning movie ID column
    movie_id_mapping = dict(zip(movies_as_arms, np.arange(len(movies_as_arms))))
    movie_id_col = []
    for movie in list(chosen_data['MovieID']):
        movie_id_col.append(movie_id_mapping[movie])

    chosen_data = chosen_data.assign(movie_col=movie_id_col)

    # Retain top % as train data and the rest for testing (to simulate real surveys)
    cnt = 0
    users_for_train = []
    for i in range(len(reverse_sorted_movies)):
        cnt += reverse_sorted_movies[i][0]
        users_for_train.append(reverse_sorted_movies[i][1])
        if cnt > train_test_ratio * chosen_data.shape[0]:
            break

    train_data = chosen_data[chosen_data['UserID'].isin(users_for_train)]
    all_users = set(chosen_data['UserID'])
    users_for_test = all_users.difference(users_for_train)
    test_data = chosen_data[chosen_data['UserID'].isin(users_for_test)]

    pathlib.Path('movies').mkdir(parents=False, exist_ok=True)

    pickle.dump(train_data, open('movies/train_data_usercount', 'wb'))
    pickle.dump(test_data, open('movies/test_data_usercount', 'wb'))

    print(f"Train-test split complete with a train-test ratio of: {train_test_ratio}")


def movie_data_process():
    train_data = pd.read_pickle('movies/train_data_usercount')
    test_data = pd.read_pickle('movies/test_data_usercount')

    movie_tables = defaultdict(np.ndarray)

    for table_movie in range(50):
        movie_tables[table_movie] = -np.ones((5, 50))

        for rating in range(1, 6):
            users = set(train_data[(train_data['movie_col'] == table_movie) &
                                   (train_data['Rating'] == rating)]['UserID'])

            movie_tables[table_movie][:, table_movie] = np.arange(1, 6)  # Initialize the table reference column

            for movie in np.delete(np.arange(0, 50), table_movie):
                d = train_data[(train_data['movie_col'] == movie) &
                               (train_data['UserID'].isin(users))]

                if d['Rating'].shape[0] != 0:
                    movie_tables[table_movie][rating - 1, movie] = np.average(
                        d['Rating'])  # Note: Use 'rating-1' to index
                else:
                    movie_tables[table_movie][rating - 1, movie] = 5.

    pickle.dump(movie_tables, open('movies/movie_tables_train.pkl', 'wb'))

    # Test data true means
    true_means = np.zeros(50)
    for movie in range(50):
        d = test_data[test_data['movie_col'] == movie]
        true_means[movie] = np.mean(d['Rating'])

    pickle.dump(true_means, open('movies/true_means_test', 'wb'))


def book_train_test_split(train_test_ratio=0.5):
    data = pd.read_csv('../data/goodreads_poetry_cropped.csv')
    data = data.drop(columns='Unnamed: 0')
    data = data[data['rating'] != 0]

    book_counts = []
    for book_id in set(data['book_id']):
        count = data[data['book_id'] == book_id].shape[0]
        book_counts.append((count, book_id))

    reverse_sorted_bookcounts = sorted(book_counts, reverse=True)
    topk_books = [book_id for (count, book_id) in reverse_sorted_bookcounts[:25]]
    book_id_mapping = dict(zip(topk_books, np.arange(len(topk_books))))

    chosen_data = data[data['book_id'].isin(topk_books)]

    # assigning book ID column
    id_assigned = []
    for book in chosen_data['book_id']:
        id_assigned.append(book_id_mapping[book])

    chosen_data = chosen_data.assign(book_col=id_assigned)

    # Retain top % as train data and the rest for testing (to simulate real surveys)
    user_counts = chosen_data['user_id'].value_counts().to_dict()
    reverse_sorted_users = sorted([(user_counts[user], user) for user in user_counts], reverse=True)

    cnt = 0
    users_for_train = []
    for i in range(len(reverse_sorted_users)):
        cnt += reverse_sorted_users[i][0]
        users_for_train.append(reverse_sorted_users[i][1])
        if cnt > train_test_ratio * chosen_data.shape[0]:
            break

    train_data = chosen_data[chosen_data['user_id'].isin(users_for_train)]
    all_users = set(chosen_data['user_id'])
    users_for_test = all_users.difference(users_for_train)
    test_data = chosen_data[chosen_data['user_id'].isin(users_for_test)]

    pathlib.Path('books').mkdir(parents=False, exist_ok=True)

    pickle.dump(train_data, open('books/train_data_usercount', 'wb'))
    pickle.dump(test_data, open('books/test_data_usercount', 'wb'))

    print(f"Train-test split complete with a train-test ratio of: {train_test_ratio}")


def book_data_process():
    train_data = pd.read_pickle('books/train_data_usercount')
    test_data = pd.read_pickle('books/test_data_usercount')

    book_tables = defaultdict(np.ndarray)

    for table_book in range(0, 25):
        book_tables[table_book] = -np.ones((5, 25))

        for rating in range(1, 6):
            users = set(train_data[(train_data['book_col'] == table_book) & \
                                   (train_data['rating'] == rating)]['user_id'])

            book_tables[table_book][:, table_book] = np.arange(1, 6)  # Initialize the table reference column

            for book in np.delete(np.arange(0, 25), table_book):
                d = train_data[(train_data['book_col'] == book) & \
                               (train_data['user_id'].isin(users))]

                if d['rating'].shape[0] < 5:  # If data is missing, pad it
                    book_tables[table_book][rating - 1, book] = 5.
                else:
                    book_tables[table_book][rating - 1, book] = np.average(d['rating'])

    pickle.dump(book_tables, open('books/book_tables_train.pkl', 'wb'))

    # Test data true means
    true_means = np.zeros(25)
    for book in range(25):
        d = test_data[test_data['book_col'] == book]
        true_means[book] = np.mean(d['rating'])

    pickle.dump(true_means, open('books/true_means_test', 'wb'))


def main(args):
    args = parse_arguments()
    if args.process_movielens:
        movielens_preproc()
        # genre exps
        genre_train_test_split(args.train_test_ratio)
        genre_data_process()

        # movie exps
        movie_train_test_split(args.train_test_ratio)
        movie_data_process()

    if args.process_goodreads:
        book_train_test_split(args.train_test_ratio)
        book_data_process()


if __name__ == '__main__':
    main(sys.argv)

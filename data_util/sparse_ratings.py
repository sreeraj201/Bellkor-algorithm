import os,json
import pandas as pd
from sklearn.model_selection import train_test_split

def create_sparseratings(directoryname='../Data/'):
    """
        Create sparse ratings and timestamps
        Returns 
            ratings_dict : for training, key=(user,movie),val=ratings
            time_dict : for training, key=(user,movie),val=time
            validratings_dict : ratings dict for parameter tuning, key=(user,movie),val=ratings
            validtime_dict : time dict for parameter tuning, key=(user,movie),val=time
            testratings_dict : ratings dict for testing, key=(user,movie),val=ratings
            testtime_dict : time dict for testing, key=(user,movie),val=time
    """
    # create dictionary for ratings and timestamp
    df_ratings = pd.read_csv(directoryname + 'ratings.csv',dtype = 'str')
    df_ratings = df_ratings.set_index(['userId','movieId'])
    rest_set,test_set = train_test_split(df_ratings,test_size = 0.1)
    training_set,valid_set = train_test_split(rest_set,test_size = 0.2)

    ratings_dict = training_set['rating'].to_dict()
    time_dict = training_set['timestamp'].to_dict()
    validratings_dict = valid_set['rating'].to_dict()
    validtime_dict = valid_set['timestamp'].to_dict()
    testratings_dict = test_set['rating'].to_dict()
    testtime_dict = test_set['timestamp'].to_dict()

    ratings_dict = {str(k) : v for k,v in ratings_dict.iteritems()}
    time_dict = {str(k) : v for k,v in time_dict.iteritems()}
    validratings_dict = {str(k) : v for k,v in validratings_dict.iteritems()}
    validtime_dict = {str(k) : v for k,v in validtime_dict.iteritems()}
    testratings_dict = {str(k) : v for k,v in testratings_dict.iteritems()}
    testtime_dict = {str(k) : v for k,v in testtime_dict.iteritems()}

    return ratings_dict,time_dict,validratings_dict,validtime_dict,testratings_dict,testtime_dict


def create_moviedict(directoryname='../Data/'):
    """
        Create a dictionary for movies with movieId as keys and 
        name and year of release and genre as values
    """
    df_movies = pd.read_csv(directoryname + 'movies.csv',index_col = ['movieId'])
    df_movies.index = df_movies.index.map(unicode)
    df_movies['Year'] = df_movies['title'].map(lambda x : None if '(' not in x else x.split('(')[-1].replace(')',""))
    df_movies['name'] = df_movies['title'].map(lambda x : x.split(',')[0] if '(' not in x else x.split('(')[0])
    movies_dict = df_movies[['name','genres','Year']].to_dict(orient = 'index')

    return movies_dict 

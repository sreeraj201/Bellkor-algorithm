import numpy as np
import os,json,random
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.externals import joblib
import pandas as pd

from sparse_ratings import create_moviedict


class Content_based:
    """
        Content based recommender
    """
    def __init__(self,ratings_dict,validratings_dict):
        """
            Initialization
        """

        self.ratings_dict = ratings_dict
        self.validratings_dict = validratings_dict

        print 'Initializing parameters ........'

        # convert string keys back to tuple
        user_list = []

        for i in self.ratings_dict.keys():
            user = i[1:-1].split(',')[0].replace("'","")
            movie = i[1:-1].split(',')[1].replace("'","")
            movie = movie.replace(" ","")

            self.ratings_dict[(user,movie)] = float(self.ratings_dict.pop(i))
            user_list.append(user)

        for i in self.validratings_dict.keys():
            user = i[1:-1].split(',')[0].replace("'","")
            movie = i[1:-1].split(',')[1].replace("'","")
            movie = movie.replace(" ","")

            self.validratings_dict[(user,movie)] = float(self.validratings_dict.pop(i))
            user_list.append(user)

        # movie dict movieno : moviename,year of release ,genre
        movie_dict = create_moviedict()
        # users list
        self.user_list = list(set(user_list))
        # movies list
        self.movie_list = movie_dict.keys()

        # create dataframe
        movie_df = pd.DataFrame.from_dict(movie_dict,orient = 'index')
        temp_df = movie_df['genres'].apply(lambda x : x.split('|'))
        temp_df = pd.get_dummies(temp_df.apply(pd.Series).stack()).sum(level = 0)

        movie_df = pd.merge(movie_df,temp_df,left_index = True, right_index = True)

        movie_df['Year'] = movie_df['Year'].apply(lambda x : x.rsplit('-',1)[0] if x else '1990')
        self.movie_df = movie_df.drop(['genres','name'],axis = 1)

        # selector dict for different modes
        self.select_ratings_dict = {'train': self.ratings_dict, 'valid': self.validratings_dict}

        print 'Initalization complete ..........' + '\n'


    def user_model(self,user,mode = 'train'):
        """
            A model for each user
            Return prediction,mean error if mode != 'train'
        """

        user_movies_list = [m for u,m in self.select_ratings_dict[mode].keys() if u == user]
        user_df = self.movie_df.ix[user_movies_list]

        user_ratings_df = pd.Series(0,index = user_movies_list)

        for m in user_movies_list:
            user_ratings_df.ix[m] = self.select_ratings_dict[mode][(user,m)]

        if mode == 'train':
            clf = GradientBoostingRegressor()
            clf.fit(user_df.values,user_ratings_df.values)

            # save to file
            joblib.dump(clf,'../Data/User_models/model_' + str(user) + '.pkl')

        else:
            clf = joblib.load('../Data/User_models/model_' + str(user) + '.pkl')
            pred = clf.predict(user_df.values)

            return np.mean((pred - user_ratings_df.values)**2)

    def model_creator(self):
        """
            Create model for all users
        """
        print 'Start training ........'

        random.seed(0)
        user_sample = random.sample(self.user_list,100)

        for user in user_sample:
            self.user_model(user)

        print 'Finished training ......' + '\n'

        print 'Start validation .......'

        total_err = 0
        for user in user_sample:
            pred,err = self.user_model(user,mode = 'valid')
            total_err = err

        print 'Finished validation .......' + '\n'

        print 'Error is {}'.format(total_err/100)


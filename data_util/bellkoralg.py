import os,time,json
import numpy as np
from datetime import datetime
import pandas as pd
import random,math

from sparse_ratings import create_moviedict
from avg_cal import calc_tu


class Bellkor:
    """
        Bellkor's Pragmatic Chaos
    """
    def __init__(self,ratings_dict,time_dict,validratings_dict,validtime_dict):
        """
            Initialization
            Input : 
                ratings_dict : key=(user,movie),val=ratings
                time_dict : key=(user,movie),val=time 
                validratings_dict : used for validation
                validtime_dict : used for validation
        """

        self.ratings_dict = ratings_dict
        self.time_dict = time_dict
        self.validratings_dict = validratings_dict
        self.validtime_dict = validtime_dict

        print 'Initializing parameters ........'

        # convert string keys back to tuple
        user_list = []
        # global mean
        self.global_mean = -1
        for num,i in enumerate(self.ratings_dict.keys()):
            user = i[1:-1].split(',')[0].replace("'","")
            item = i[1:-1].split(',')[1].replace("'","")
            if self.global_mean == -1:
                self.global_mean = float(self.ratings_dict[i])
            else:
                self.global_mean += (float(self.ratings_dict[i]) - self.global_mean)/num

            self.ratings_dict[(user,item)] = float(self.ratings_dict.pop(i))
            self.time_dict[(user,item)] = int(self.time_dict.pop(i))
            user_list.append(user)

        for i in self.validratings_dict.keys():
            user = i[1:-1].split(',')[0].replace("'","")
            item = i[1:-1].split(',')[1].replace("'","")
            self.validratings_dict[(user,item)] = float(self.validratings_dict.pop(i))
            self.validtime_dict[(user,item)] = int(self.validtime_dict.pop(i))
            user_list.append(user)


        # movie dict with movieno, moviename and year of release 
        self.movie_dict = create_moviedict()
        # users list
        self.user_list = list(set(user_list))
        # movies list
        self.movie_list = self.movie_dict.keys()
        # time list
        time_list = list(self.time_dict.values()) + list(self.validtime_dict.values())

        self.time_list = list(set(time_list))
        self.start_time = min(self.time_list)
        self.end_time = max(self.time_list)

        self.params = {}
        self.params['b_u'] = {}
        self.params['alpha_u'] = {}
        self.params['c_u'] = {}
        self.params['b_i'] = {}

        # matrix factorization
        self.params['q'] = {}
        self.params['p'] = {}
        self.params['alpha_p'] = {}
        k = 20

        for user in self.user_list:
            self.params['b_u'][user] = 0
            self.params['alpha_u'][user] = 0
            self.params['c_u'][user] = 1
            self.params['p'][user] = np.random.rand(k) * 1e-2
            self.params['alpha_p'][user] = np.random.rand(k) * 1e-2

        self.params['b_ut'] = pd.DataFrame(0,index = self.user_list,columns = self.time_list)
        self.params['c_ut'] = pd.DataFrame(0,index = self.user_list,columns = self.time_list)

        temp = pd.DataFrame(0,index = self.movie_list, columns = range(30)) 
        summer = pd.Series(range(30),index = range(30))
        self.params['b_ibin'] = temp.add(summer,axis = 1)


        self.params['b_ibin'] = self.params['b_ibin'].astype('float')/100

        for item in self.movie_list:
            self.params['b_i'][item] = 0
            self.params['q'][item] = np.random.rand(k) * 1e-2

        # learning rates
        self.learning_rate = {}
        self.learning_rate['b_u'] = 5e-3
        self.learning_rate['alpha_u'] = 1e-4
        self.learning_rate['b_i'] = 2e-2
        self.learning_rate['c_u'] = 5e-3
        self.learning_rate['b_ut'] = 2e-2
        self.learning_rate['c_ut'] = 2e-2
        self.learning_rate['b_ibin'] = 1e-4
        self.learning_rate['p'] = 5e-3
        self.learning_rate['q'] = 5e-3
        self.learning_rate['alpha_p'] = 1e-4

        # regularization
        self.reg = {}
        self.reg['b_u'] = 3e-5
        self.reg['alpha_u'] = 5e-3
        self.reg['b_i'] = 3e-3
        self.reg['c_u'] = 3e-5
        self.reg['b_ut'] = 5e-2
        self.reg['c_ut'] = 5e-2
        self.reg['b_ibin'] = 1e-4
        self.reg['p'] = 5e-4
        self.reg['q'] = 5e-4
        self.reg['alpha_p'] = 1e-4

        # others
        self.t_u = calc_tu()

        # selector dict for different modes
        self.select_ratings_dict = {'train': self.ratings_dict, 'valid': self.validratings_dict}
        self.select_time_dict = {'train' : self.time_dict, 'valid' : self.validtime_dict}

        print 'Initalization complete ..........' + '\n'

    def main_alg(self,mode = 'train',seed_val = 0,sample_count = 100):
        """
            Main algorithm
            Input : 
                mode : ['train','valid']
                seed_val : for random seed
                sample_count : sample size for training
            Returns cost and error; output if mode != 'train'
        """
        cost = 0
        total_error = 0
        num = 0

        random.seed(seed_val)

        # randomly sample keys
        key_list = self.select_ratings_dict[mode].keys()
        sample_keys = random.sample(key_list,sample_count)

        output = {}

        # for key,rating in self.ratings_dict.iteritems():
        for count,key in enumerate(sample_keys):

            if count%20==0 and mode == 'train':
                print 'trained {} users'.format(count)

            rating = self.select_ratings_dict[mode][key]
            user,movie = key
            # remove whitespace
            movie = movie.replace(" ","")
            # calc time difference with avg
            delta = (self.select_time_dict[mode][key] - self.t_u.loc[user]['Avg_Date'])
            if delta < 0 :
                sign = -1
            else:
                sign = 1

            # rescale
            delta = abs(delta)/1e7
            # deviation from mean
            dev = sign * math.pow(delta,0.4)
            # calc bin values;700 consecutive days = 30 bins
            binval = (self.select_time_dict[mode][key] - self.start_time)/7e+7
            binval = int(binval)
            # temp variables
            b_u,alpha_u,b_i,c_u = self.params['b_u'][user],self.params['alpha_u'][user],self.params['b_i'][movie],self.params['c_u'][user]
            b_ut,c_ut = self.params['b_ut'].loc[user,self.select_time_dict[mode][key]],self.params['c_ut'].loc[user,self.select_time_dict[mode][key]]
            b_ibin = self.params['b_ibin'].loc[movie,binval]
            alpha_p = self.params['alpha_p'][user]
            p = self.params['p'][user]  
            q = self.params['q'][movie]
            P = p + alpha_p * dev
            # output
            out = self.global_mean + b_u + (alpha_u*dev) + b_ut + (b_i + b_ibin) * (c_u + c_ut) + np.dot(q.T,P)
            output[key] = out
            # cost cal
            err =  rating - out

            # for parameter tuning
            # if abs(err) > 10:
            #     print err
            #     print 'b_u = {}, alpha_u = {}, dev = {},b_ut = {}, b_i = {}, b_ibin = {},  c_u = {}, c_ut = {},dev = {}'.format(b_u,alpha_u,dev,b_ut,b_i,b_ibin,c_u,c_ut,dev)

            # else:
            num += 1
            total_error += abs(err)

            cost += err**2
            # reg terms
            cost += (self.reg['b_u']*(b_u**2) + self.reg['alpha_u']*(alpha_u**2) + self.reg['b_ut']*(b_ut**2) + 
                    self.reg['b_i']*(b_i**2) + self.reg['b_ibin']*(b_ibin**2) + self.reg['c_u']*(c_u-1)**2 + 
                    self.reg['c_ut']*(c_ut**2) + self.reg['p']*(p**2) + self.reg['q']*(q**2) + self.reg['alpha_p']*(alpha_p**2))

            if mode == 'train':
                # gradient terms
                grads = {}
                grads['b_u'] = 2*err*(-1+ 2*self.reg['b_u']*b_u) 
                grads['alpha_u'] = 2*err*(-dev + 2*self.reg['alpha_u']*alpha_u)
                grads['b_ut'] = 2*err*(-1 + self.reg['b_ut']*b_ut)
                grads['b_i'] = 2*err*(-c_u - c_ut  + 2*self.reg['b_i']*b_i)
                grads['b_ibin'] = 2*err*(-c_u - c_ut + 2*self.reg['b_ibin']*b_ibin)
                grads['c_u'] = 2*err*(-b_i - b_ibin + 2*self.reg['c_u']*(c_u-1))
                grads['c_ut'] = 2*err*(-b_i - b_ibin + 2*self.reg['c_ut']*c_ut)
                grads['p'] = 2*err*(q + 2*self.reg['p']*p) 
                grads['q'] = 2*err*(p+alpha_p*dev+2*self.reg['q']*q)
                grads['alpha_p'] = 2*err*(q*dev + self.reg['alpha_p']*alpha_p)

                self.update(grads,user,movie,key,binval)


        print 'num is {}'.format(num)

        if mode == 'train':
            return cost/num,total_error/num

        else:
            return output,cost/num,total_error/num


    def update(self,grads,user,movie,key,binval):
        """
            Update the parameters (only during training)
        """
        self.params['b_u'][user] -= self.learning_rate['b_u'] * grads['b_u']
        self.params['alpha_u'][user] -= self.learning_rate['alpha_u'] * grads['alpha_u']
        self.params['b_i'][movie] -= self.learning_rate['b_i'] * grads['b_i']
        self.params['c_u'][user] -= self.learning_rate['c_u'] * grads['c_u']
        self.params['b_ut'].loc[user,self.time_dict[key]] -= self.learning_rate['b_ut'] * grads['b_ut']
        self.params['c_ut'].loc[user,self.time_dict[key]] -= self.learning_rate['c_ut'] * grads['c_ut']
        self.params['b_ibin'].loc[movie,binval] -= self.learning_rate['b_ibin'] * grads['b_ibin']
        self.params['p'][user] -= self.learning_rate['p'] * grads['p']
        self.params['q'][movie] -= self.learning_rate['q'] * grads['q']
        self.params['alpha_p'][user] -= self.learning_rate['alpha_p'] * grads['alpha_p']


    def train(self):
        """
            Runs the algorithm
        """
        print "Start Training ........."
        for epoch in range(10):
            print 'epoch num : {}'.format(epoch)
            cost,error = self.main_alg(seed_val = int(epoch/5))
            print 'Error at this epoch is {}'.format(error)

        print 'Finished Training ..........' + '\n'


    def predict(self):
        """
            Predict ratings
        """
        output,cost,error = self.main_alg(mode = 'valid',seed_val = 1000)

        print "Val error is {}".format(error)

        print 'global_mean is {}'.format(self.global_mean)

        # sample_keys = random.sample(output.keys(),20)
        # for key in sample_keys:
        #     print output[key],self.validratings_dict[key]

        return output

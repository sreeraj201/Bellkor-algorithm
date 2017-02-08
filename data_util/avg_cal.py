import json
import pandas as pd
from datetime import datetime
from datetime import timedelta

def calc_tu(directoryname = '../Data/'):
    """
        Calculate tu for each user
    """
    with open(directoryname + 'time_dict.json')as f:
        time_dict = json.load(f)

    tu_dict = {}

    for i,val in time_dict.iteritems():
        user = i[1:-1].split(',')[0].replace("'","")
        item = i[1:-1].split(',')[1].replace("'","")
        if user not in tu_dict:
            tu_dict[user] = []
        # val = datetime.strptime(val,'%Y-%m-%d')
        tu_dict[user].append(val)

    for k in tu_dict.keys():
        tu_dict[k] = list(set(tu_dict[k]))
        tu_dict[k] = [int(t) for t in tu_dict[k]]
        mean_date = sum(tu_dict[k])/len(tu_dict[k])
        tu_dict[k] = mean_date

    df = pd.DataFrame.from_dict(tu_dict,orient = 'index')
    df.columns = ['Avg_Date']
    df.index.name = 'userId'
    df.index = df.index.map(unicode)

    return df

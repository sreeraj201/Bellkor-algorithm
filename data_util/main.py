import numpy as np

from sparse_ratings import *
from bellkoralg import *
from content_based import *


def main():
    """
        Main point of entry
    """
    # sparse ratings
    json1,json2,json3,json4,json5,json6 = create_sparseratings()

    with open('../Data/ratings_dict.json','w') as f:
        json.dump(json1,f,indent = 4)
    with open('../Data/time_dict.json','w') as f:
        json.dump(json2,f,indent = 4)
    with open('../Data/validratings_dict.json','w') as f:
        json.dump(json3,f,indent = 4)
    with open('../Data/validtime_dict.json','w') as f:
        json.dump(json4,f,indent = 4)
    with open('../Data/testratings_dict.json','w') as f:
        json.dump(json5,f,indent = 4)
    with open('../Data/testtime_dict.json','w') as f:
        json.dump(json6,f,indent = 4)

    # bellkor
    print "Loading dataset ........"
    # ratings dict with keys as a tuple of user id and movie no
    with open('../Data/ratings_dict.json') as f:
        ratings_dict = json.load(f)
    # time dict with values representing date of rating
    with open('../Data/time_dict.json') as f:
        time_dict = json.load(f)

    # validation ratings
    with open('../Data/validratings_dict.json') as f:
        validratings_dict = json.load(f)
    # validation time
    with open('../Data/validtime_dict.json') as f:
        validtime_dict = json.load(f)
    print "Loading Complete ........" + '\n'

    inst = Bellkor(ratings_dict,time_dict,validratings_dict,validtime_dict)
    inst.train()
    output = inst.predict()

    # content based
    print "Loading dataset ........."
    # ratings dict with keys as a tuple of user id and movie no
    with open('../Data/ratings_dict.json') as f:
        ratings_dict = json.load(f)
    # validation ratings
    with open('../Data/validratings_dict.json') as f:
        validratings_dict = json.load(f)
    print "Loading Complete ........" + '\n'

    c = Content_based(ratings_dict,validratings_dict)
    c.model_creator()

if __name__ == "__main__":
    main()

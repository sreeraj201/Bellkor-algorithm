import simplejson as json

from textblob import TextBlob

def json_todict(directoryname = '',filename = 'review.json'):
    """
        Converts json file to dictionary
    """
    movies_list = list()
    with open(directoryname + filename) as f:
        movies_list = json.load(f)

    return movies_list

def sentiment_analyzer(movies_list):
    """
        Analyzes the sentiment in the dictionary
    """
    movies_dict = dict()
    for movie in movies_list:
        movies_dict['movieid'] = movie['movieid']
        movies_dict['moviename'] = movie['moviename']
        reviews_list = movie['review']
        polarity = 0
        for review in reviews_list:
            fullreview = TextBlob(review['fullreview'])
            polarity += fullreview.sentiment.polarity

        polarity /= len(reviews_list)
        movies_dict['polarity'] = polarity
        yield movies_dict


if __name__ == '__main__':
    movies_list = json_todict()
    movies_dict = sentiment_analyzer(movies_list)
    a = json.dumps(movies_dict,iterable_as_array = True)
    with open('moviepolarity.json','wb') as f:
        f.write(a)

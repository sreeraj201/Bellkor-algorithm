# -*- coding: utf-8 -*-
import scrapy
import pandas as pd
from itertools import izip
import re

from scrapy.loader import ItemLoader
from imdbreviews.items import ImdbreviewsItem


class ReviewSpider(scrapy.Spider):
    name = "review"
    allowed_domains = ["www.imdb.com"]

    def __init__(self):
        """
            List of urls in csv file
        """
        self.df = pd.read_csv('/media/sreeraj/Work/Code/Kaggle/grouplens/Data/links.csv',header = 0)
        self.start_urls = self.df['imdbId'].map(lambda x : "http://www.imdb.com/title/tt"+ str(x) +'/reviews?start=0').tolist()
        # for testing
        # self.start_urls  = ['http://www.imdb.com/title/tt0114709/reviews?start=0','http://www.imdb.com/title/tt0113497/reviews?start=0']

        return

    def parse_review(self,response):
        for oneliner,fullreview in izip(response.css('h2::text').extract(),response.css('div+ p::text').extract()):
            oneliner,fullreview = oneliner.encode('ascii','ignore'),fullreview.encode('ascii','ignore')
            yield{
                    "onelinereview" : oneliner.replace('"',''),
                    "fullreview": fullreview.replace('"','')
                    }


    def parse(self, response):
        movieid = re.search('http://www.imdb.com/title/tt(\w*)/reviews.*',response.url)
        movieid = self.df['movieId'].loc[self.df['imdbId'] == int(movieid.group(1))].values[0]
        movie = ItemLoader(item = ImdbreviewsItem(), response = response)
        movie.add_value('movieid',movieid)
        movie.add_css('moviename','.main::text')
        movie.add_value('review', self.parse_review(response))

        return movie.load_item()


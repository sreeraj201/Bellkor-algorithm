# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ImdbreviewsItem(scrapy.Item):
    movieid = scrapy.Field()
    moviename = scrapy.Field()
    review = scrapy.Field()

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.selector import Selector
from scrapy_splash import SplashRequest
import re
import pprint
import pickle
import time
import json
import requests
import io

class CommentSpider(scrapy.Spider):
  # this spider scrapes a single article within the domain zeit.de
  name = 'people'
  urls = [['http://en.people.cn/n3/2017/1015/c90000-9279881.html']]
  file = None
  current_url = ''

  def __init__(self):
    self.file = open('output.csv', 'wb')

  def start_requests(self):
    self.file.write('cid,url,author,time,parent,votes,text\n'.encode('utf-8'))
    for url in self.urls:
      self.current_url = url[0]
      yield SplashRequest(
         url=str(url[0]),
         callback=self.get_comments,
         endpoint='render.html',
         args={'wait': 0.5},
        )

  def get_comments(self, response):
    sel = Selector(response)
    comment_url = sel.xpath("//div[@class='liuyan_box']//iframe[@src]/@src").extract()
    if (len(comment_url) > 0):
      nextLink = comment_url[0]
      request = SplashRequest(url=nextLink, callback = self.parse_comments, endpoint='render.html', args={'wait': 0.5})
      request.meta['url'] = comment_url[0]
      yield request 

  def parse_comments(self, response):
    sel = Selector(response)
    load_more = sel.xpath("//div[@class='load-more']/a").extract()
    scriptt = sel.xpath("//script[@type='text/json']/text()").extract()
    data = json.loads(scriptt[1])
    thread_id = data['response']['thread']['id']
    cursor = "0:0:0"
    while (True):
      proper_url = "https://disqus.com/api/3.0/threads/listPostsThreaded?limit=100&thread="+str(thread_id)+"&forum=enpeople&order=popular&api_key=E8Uh5l5fHZ6gD8U3KycjAIAk46f68Zw7C6eW8WSjZvCLXebZ7p0r1yrYDrLilk2F&cursor="
      appended_url = proper_url + cursor
      res = requests.get(appended_url)
      data = res.json()
      for post in data['response']:
        line = "\"" + str(post['id']).encode('utf-8') + "\",\"" + \
          self.current_url.encode('utf-8') + "\",\"" + \
          post['author']['name'].replace('"', '""').encode('utf-8') + "\",\"" + \
          str(post['createdAt']).encode('utf-8') + "\",\"" + \
          str(post['parent']).encode('utf-8') + "\",\"" + \
          str(post['likes']).encode('utf-8') + "\",\"" + \
          post['raw_message'].replace('"', '""').encode('utf-8') + "\"\n"
        self.file.write(line)
      if data['cursor']['hasNext']:
        cursor = data['cursor']['next']
      else:
        break

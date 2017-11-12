import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.selector import Selector
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import re
import pprint
import pickle
import time
import json
import requests
import io

class CommentSpider(CrawlSpider):
  # this spider scrapes a single article within the domain zeit.de
  name = 'people'
  allowed_domains = ['en.people.cn']
  #urls = [['http://en.people.cn/review/']]
  start_urls = ['http://en.people.cn/review/']
  file = None
  disqus_identifier_regex = re.compile('(?<=disqus_identifier = \')[0-9]+')
  rules = (
    Rule (LinkExtractor(allow_domains = ['en.people.cn']), callback="parse_comments", follow=True),
  )

  def __init__(self, *a, **kw):
    super(CommentSpider, self).__init__(*a, **kw)
    self.start_urls= []
    stringstart = 'http://en.people.cn/review/'
    for j in range(1, 31):
      for k in range(2015, 2017):
        for x in range(1, 12):
          ns = stringstart + str(k)
          if x < 10:
            ns += '0'
          ns += str(x)
          if j < 10:
            ns += '0'
          ns += str(j) + '.html'
          self.start_urls.append(ns)
    self.file = open('output.csv', 'wb')
    self.file.write('cid,url,author,time,parent,likes,dislikes,text\n'.encode('utf-8'))

  def parse_comments(self, response):
    thread_id_possibilities = self.disqus_identifier_regex.findall(response.body)
    if thread_id_possibilities:
      thread_id = thread_id_possibilities[0]
    else:
      return
    cursor = "0:0:0"
    while (True):
      proper_url = "https://disqus.com/api/3.0/threads/listPostsThreaded?limit=100&thread:ident="+str(thread_id)+"&forum=enpeople&order=popular&api_key=E8Uh5l5fHZ6gD8U3KycjAIAk46f68Zw7C6eW8WSjZvCLXebZ7p0r1yrYDrLilk2F&cursor="
      appended_url = proper_url + cursor
      res = requests.get(appended_url)
      data = res.json()
      for post in data['response']:
        line = "\"" + str(post['id']).encode('utf-8') + "\",\"" + \
          response.url.encode('utf-8') + "\",\"" + \
          post['author']['name'].replace('"', '""').encode('utf-8') + "\",\"" + \
          str(post['createdAt']).encode('utf-8') + "\",\"" + \
          str(post['parent']).encode('utf-8') + "\",\"" + \
          str(post['likes']).encode('utf-8') + "\",\"" + \
          str(post['dislikes']).encode('utf-8') + "\",\"" + \
          post['raw_message'].replace('"', '""').encode('utf-8') + "\"\n"
        self.file.write(line)
      if data['cursor']['hasNext']:
        cursor = data['cursor']['next']
      else:
        break

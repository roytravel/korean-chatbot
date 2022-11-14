import requests
from collections import namedtuple
from typing import Text, List, Dict, Any
from events import Slotset
from bs4 import BeautifulSoup

class Weather:
    def __init__(self):
        pass

    def run(self, dispatcher, tracker, domain):
        raise NotImplementedError()

    def get_weather(self):
        raise NotImplementedError()

    def get_rain_prob(self):
        raise NotImplementedError()

    def get_find_dust(self):
        raise NotImplementedError()

    def get_ultra_fine_dust(self):
        raise NotImplementedError()
    
    def get_ozone(self):
        raise NotImplementedError()
    

class News:
    def __init__(self):
        pass

    def get_weather(self):
        raise NotImplementedError()

    def get_news(self):
        raise NotImplementedError()
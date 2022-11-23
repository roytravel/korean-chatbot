import os
import requests
from bs4 import BeautifulSoup

class Weather:
    def __init__(self):
        self.addr_wth = "https://weather.naver.com/today/"
        self.addr_air = "https://weather.naver.com/air/"
        self.addr_sun = "https://search.naver.com/search.naver?query="
        self.session = requests.Session()
        
    def get_weather_naver(self):
        response = self.session.get(self.addr_wth).text
        soup = BeautifulSoup(response, 'lxml')
        day_data = soup.find(class_="day_data")
        
        # msg: 성북구 안암동
        location = soup.find(class_="location_name").text 
        
        # msg: 현재 온도 17.0' / 맑음
        temperature = soup.find(class_ = "current").text.strip('\n') 
        temperature_state = soup.find(class_='weather').text
        
        # msg: 최저기온N' 최고기온M'
        lowest = day_data.findAll("span", {"class":"lowest"})[0].text
        highest = day_data.findAll("span", {"class":"highest"})[0].text
        
        # msg: 오전 강수확률N% 오후 강수확률M%
        # timeslot = day_data.findAll("span", {"class":"timeslot"})
        rainfall = day_data.findAll("span", {"class":"rainfall"})
        
        # msg: 현재 미세먼지 농도 N 좋음        
        response = self.session.get(self.addr_air).text
        soup = BeautifulSoup(response, 'lxml')
        phrase = soup.select('#content > div > div.section_right > div.card.card_dust > h2')[0].get_text()
        density = soup.find(class_="value _cnPm10Value").text 
        state = soup.find(class_="grade _cnPm10Grade").text
        
        # msg: 자외선 지수 좋음
        # category = soup.find(class_="tit").text
        # state = soup.find(class_="level_dsc").text
        
        message = f"오늘 {location} 날씨 알려드릴게요. {temperature}로 {temperature_state}입니다. {lowest}이며 {highest}입니다. 또 오전 {rainfall[0].text}이며 오후 {rainfall[1].text}입니다. {phrase}는 {density}이며 {state}입니다. "
        return message

    def get_weather_OpenWeatherMap(self):
        raise NotImplementedError()

class News:
    def __init__(self) -> None:
        super().__init__()
        
class Translate:
    def translate_ko_en(text):
        data = {'text':text, 'source':'ko','target':'en'}
        client_id = os.getenv("CLIENT_ID")
        client_secret = os.getenv("CLIENT_SECRET")
        url = "https://openapi.naver.com/v1/papago/n2mt"
        header = {"X-Naver-Client-Id":client_id,"X-Naver-Client-Secret":client_secret}
        response = requests.post(url, headers=header, data= data)
        if response.status_code == 200:
            translated_text = response.json()
            return translated_text['message']['result']['translatedText']
        else:
            return f'[!] Error: {response.status_code}'
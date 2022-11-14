from collections import namedtuple
from typing import Text, List, Dict, Any

KobotMessage = namedtuple("KobotMessage", "text data")

class Distpacher:
    """ 사용자에게 메시지를 반환 """
    def __init__(self):
        pass
    
    async def utter_response(self):
        pass
    
    async def utter_mssage(self):
        pass 
from collections import namedtuple
from typing import Text, List, Dict, Any, Optional
# from events import Slotset
from abc import ABCMeta, abstractmethod
from core.state_tracker import DialogueStateTracker


class Action(metaclass=ABCMeta):
    @abstractmethod
    async def run(self, dispatcher, tracker, domain) -> List[Dict[Text, Any]]:
        raise NotImplementedError() # or pass


class Dispatcher:
    """ 응답 메시지 반환 """
    def send_message_to_user(message):
        return message
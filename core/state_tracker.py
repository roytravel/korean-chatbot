import os
import sys
import json
from typing import List, Tuple, Any, Text, Dict, Optional
from utils.decorators import data
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class Domain:
    @classmethod
    def load_file(cls, filename):
        return cls.from_json(cls.read_file(filename))
        
    @classmethod
    def from_json(cls, jsondata: json.load):
        intents = jsondata['intents']
        entities = jsondata['entities']
        slots = jsondata['slots']
        templates = jsondata['templates']
        actions = jsondata['actions']
        return intents, entities, slots, templates, actions
    
    @classmethod
    def read_file(cls, filename: str) -> json.load:
        with open(filename, mode="rt", encoding="utf-8") as f:
            return json.load(f)
    
@data
class DialogueStateTracker(Domain):
    def __init__(self):
        domain = Domain.load_file(self.DOMAIN_FILENAME)
        self.intents = domain[0]
        self.entities = domain[1]
        self.slots = domain[2]
        self.templates = domain[3]
        self.actions = domain[4]
    
    def fill_slot(self, intent, entity):
        return intent, entity
        
    def get_slot(self, key) -> Optional[Any]:
        return self.slot[key]
    
    def add_slot(self, slot: Dict[Text, Any]) -> None:
        self.slots.append(slot)
        
    def __save_slot(self) -> None:
        raise NotImplementedError()
from typing import List, Tuple, Any, Text, Dict, Optional

class DialogueStateTracker:
    def __init__(self, slots: Dict[Text, Any]):
            self.slots = list(slots)
        
    def get_slot(self, key) -> Optional[Any]:
        return self.slot[key]
    
    def add_slot(self, slot: Dict[Text, Any]) -> None:
        self.slots.append(slot)
    
    def __save_slot(self) -> None:
        raise NotImplementedError()

    
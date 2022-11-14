from typing import Text, List, Dict, Any, Optional

EventType = Dict[Text, Any]

def SlotSet(key: Text, value: Any = None, timestamp: Optional[float] = None) -> EventType:
    return {"event": "slot", "timestamp": timestamp, "name": key, "value": value}
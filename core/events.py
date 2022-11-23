from typing import Text, List, Dict, Any, Optional

def SlotSet(key: Text, value: Any = None, timestamp: Optional[float] = None) -> Dict[Text, Any]:
    return {"event": "slot", "timestamp": timestamp, "name": key, "value": value}
from core.knowledge_base.api import Weather

def define_action(intent):
    """
    return
        0: static action
        1: dynamic action
    """
    if intent in [0, 1, 2, 3]:
        action_type = 1
    else:
        action_type = 0
    
    return intent, action_type

    
def trigger_action(intent: int, action_type: int) -> str:
    """ action_type: 1(dynamic), 0(static) """
    if action_type == 1:
        W = Weather()
        if intent == 0:
            result = W.get_weather_naver()
            return result
    elif action_type == 0:
        return NotImplementedError()        
    else:
        raise ValueError()
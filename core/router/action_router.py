from core.router.lobe_actions_map import LOBE_ACTION_MAP
from core.router.actions import BrainAction

def route_action(lobe: str, confidence: float):
    """
    Decide what the system should DO next based on lobe + confidence
    """

    # Low confidence
    if confidence < 0.45:
        return {
            "action": BrainAction.ASK_CLARIFICATION,
            "reason": "Low confidence in lobe classification"
        }

    # Normal confidence 
    actions = LOBE_ACTION_MAP.get(lobe, [])

    return {
        "action": actions[0],
        "alternatives": actions[1:]
    }

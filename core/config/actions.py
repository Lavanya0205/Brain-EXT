from enum import Enum

class BrainAction(str, Enum):
    PLAN = "plan"
    EXPLAIN = "explain"
    VISUALIZE = "visualize"
    RECALL = "recall"
    ASK_CLARIFICATION = "ask_clarification"

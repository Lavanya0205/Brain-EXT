from enum import Enum

class BrainAction(str, Enum):
    PLAN = "plan"
    DECIDE = "decide"
    STRUCTURE = "structure"

    EXPLAIN = "explain"
    RECALL = "recall"
    SUMMARIZE = "summarize"

    ORGANIZE = "organize"
    RELATE = "relate"
    MAP_CONCEPTS = "map_concepts"

    VISUALIZE = "visualize"
    IMAGINE = "imagine"
    DIAGRAM = "diagram"

    STORE_MEMORY = "store_memory"
    ASK_CLARIFICATION = "ask_clarification"

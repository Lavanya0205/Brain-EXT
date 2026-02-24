from core.config.actions import BrainAction

def decide_action(lobe: str, confidence: float, memory_context = None) -> BrainAction:
    if confidence < 0.45:
        return BrainAction.ASK_CLARIFICATION

    if lobe == "frontal":
        return BrainAction.PLAN

    if lobe == "temporal":
        return BrainAction.EXPLAIN

    if lobe == "parietal":
        return BrainAction.RECALL

    if lobe == "occipital":
        return BrainAction.VISUALIZE

    return BrainAction.ASK_CLARIFICATION

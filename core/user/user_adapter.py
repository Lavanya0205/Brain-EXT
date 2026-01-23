from core.user.user_model import user_model

def adapt_action(original_action, confidence):
    """
    Modify system behavior based on user history
    """
    profile = user_model.summary()

    # If user is often confused → slow down
    if profile["confusion_level"] >= 3 and confidence < 0.6:
        return "ask_clarification"

    # If user prefers explanation → respect it
    if profile["preferred_action"] == "explain":
        return "explain"

    return original_action

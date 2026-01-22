class UserModel:
    def __init__(self):
        self.dominant_lobe = None
        self.preferred_action = None
        self.confusion_level = 0
        self.total_interactions = 0

    def update(self, lobe, action, confidence):
        self.total_interactions += 1

        # Update confusion
        if confidence < 0.5:
            self.confusion_level += 1

        # Update dominant lobe
        if not self.dominant_lobe:
            self.dominant_lobe = lobe

        # Update preferred action
        if not self.preferred_action:
            self.preferred_action = action

    def summary(self):
        return {
            "dominant_lobe": self.dominant_lobe,
            "preferred_action": self.preferred_action,
            "confusion_level": self.confusion_level,
            "total_interactions": self.total_interactions
        }


# Global user model (single-user system for now)
user_model = UserModel()

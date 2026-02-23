class UserModel:
    def __init__(self):
        self.lobe_scores = {
            "frontal": 0,
            "temporal": 0,
            "parietal": 0,
            "occipital": 0
        }

        self.action_scores = {}
        self.confusion_level = 0
        self.total_interactions = 0

    def update(self, lobe, action, confidence):
        self.total_interactions += 1

        # Track lobe frequency
        if lobe in self.lobe_scores:
            self.lobe_scores[lobe] += 1

        # Track action frequency
        if action not in self.action_scores:
            self.action_scores[action] = 0
        self.action_scores[action] += 1

        # Update confusion
        if confidence < 0.5:
            self.confusion_level += 1
        else:
            self.confusion_level = max(0, self.confusion_level - 1)

    def get_dominant_lobe(self):
        return max(self.lobe_scores, key=self.lobe_scores.get)

    def get_preferred_action(self):
        if not self.action_scores:
            return None
        return max(self.action_scores, key=self.action_scores.get)

    def summary(self):
        return {
            "dominant_lobe": self.get_dominant_lobe(),
            "preferred_action": self.get_preferred_action(),
            "confusion_level": self.confusion_level,
            "total_interactions": self.total_interactions
        }


# Global user model (single-user system for now)
user_model = UserModel()
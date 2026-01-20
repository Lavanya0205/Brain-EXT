class LongTermMemory:
    def __init__(self):
        self.data = {
            "lobe_counts": {},
            "confusion_count": 0,
            "preferred_action": None
        }

    def update_lobe(self, lobe):
        self.data["lobe_counts"][lobe] = self.data["lobe_counts"].get(lobe, 0) + 1

    def increment_confusion(self):
        self.data["confusion_count"] += 1

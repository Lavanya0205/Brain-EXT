from core.memory.short_term import ShortTermMemory
from core.memory.long_term import LongTermMemory

short_memory = ShortTermMemory()
long_memory = LongTermMemory()

def update_memory(query, lobe, action, confidence):
    short_memory.add({
        "query": query,
        "lobe": lobe,
        "action": action,
        "confidence": confidence
    })

    long_memory.update_lobe(lobe)

    if confidence < 0.5:
        long_memory.increment_confusion()

def get_memory_context():
    return {
        "short_term_memory": short_memory.get(),
        "long_term_memory": long_memory.data
    }

from core.memory.short_term import ShortTermMemory
from core.memory.long_term import LongTermMemory

short_memory = {
    "frontal": ShortTermMemory(),
    "temporal": ShortTermMemory(),
    "parietal": ShortTermMemory(),
    "occipital": ShortTermMemory()
}

long_memory = {
    "frontal": LongTermMemory(),
    "temporal": LongTermMemory(),
    "parietal": LongTermMemory(),
    "occipital": LongTermMemory()
}

def update_memory(query, lobe, action, confidence):
    short_memory[lobe].add({
        "query": query,
        "lobe": lobe,
        "action": action,
        "confidence": confidence
    })
    long_memory[lobe].update_lobe(lobe)

    if confidence < 0.5:
        long_memory[lobe].increment_confusion()

def get_memory_context(lobe=None):
    if lobe:
        return {
            "short_term_memory": short_memory[lobe].get(),
            "long_term_memory": long_memory[lobe].data
        }
    return {
        lobe_name: {
            "short_term_memory": short_memory[lobe_name].get(),
            "long_term_memory": long_memory[lobe_name].data
        }
        for lobe_name in short_memory
    }
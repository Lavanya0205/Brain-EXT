from core.router.actions import BrainAction

LOBE_ACTION_MAP = {
    "frontal": [
        BrainAction.PLAN,
        BrainAction.DECIDE,
        BrainAction.STRUCTURE
    ],

    "temporal": [
        BrainAction.EXPLAIN,
        BrainAction.RECALL,
        BrainAction.SUMMARIZE
    ],

    "parietal": [
        BrainAction.ORGANIZE,
        BrainAction.RELATE,
        BrainAction.MAP_CONCEPTS
    ],

    "occipital": [
        BrainAction.VISUALIZE,
        BrainAction.IMAGINE,
        BrainAction.DIAGRAM
    ]
}

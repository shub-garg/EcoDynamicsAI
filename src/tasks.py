# tasks.py

"""
This module defines all prediction tasks in a structured way.
Each task references one 'target' column and a list of 'features'.
The 'features' come from the categories in your EcodynamicsAI class
(e.g., 'environmental', 'plant_otus', 'plant_metrics', etc.).

Usage Example:
    from tasks import define_tasks

    # Suppose you have:
    #   framework.feature_categories = {
    #       'environmental': [...],
    #       'plant_otus': [...],
    #       'plant_metrics': [...],
    #       # etc.
    #   }
    #
    # Then you can do:
    #   tasks = define_tasks(framework.feature_categories)
    #
    # And feed 'tasks' into your training pipeline.
"""

def define_tasks(feature_categories):
    """
    Given a dictionary of feature categories, return a dictionary
    of tasks. Each task has:
      - 'target': the name of the target column
      - 'features': the list of feature columns to use

    The feature_categories argument should be something like:
        {
            'environmental': [...],
            'leaf_otus': [...],
            'root_otus': [...],
            'plant_otus': [...],
            'specific_metrics': [...],
            'plant_metrics': [...]
        }
    """

    tasks = {
        # 1. Microbiota Richness
        "Microbiota Richness (Root) with Plant OTUs": {
            "target": "richness_microbiota_root",
            "features": feature_categories['environmental'] + feature_categories['plant_otus']
        },
        "Microbiota Richness (Leaf) with Plant OTUs": {
            "target": "richness_microbiota_leaf",
            "features": feature_categories['environmental'] + feature_categories['plant_otus']
        },
        "Microbiota Richness (Root) with Plant Metrics": {
            "target": "richness_microbiota_root",
            "features": feature_categories['environmental'] + feature_categories['plant_metrics']
        },
        "Microbiota Richness (Leaf) with Plant Metrics": {
            "target": "richness_microbiota_leaf",
            "features": feature_categories['environmental'] + feature_categories['plant_metrics']
        },

        # 2. Microbiota Shannon
        "Microbiota Shannon (Root) with Plant OTUs": {
            "target": "Shannon_microbiota_root",
            "features": feature_categories['environmental'] + feature_categories['plant_otus']
        },
        "Microbiota Shannon (Leaf) with Plant OTUs": {
            "target": "Shannon_microbiota_leaf",
            "features": feature_categories['environmental'] + feature_categories['plant_otus']
        },
        "Microbiota Shannon (Root) with Plant Metrics": {
            "target": "Shannon_microbiota_root",
            "features": feature_categories['environmental'] + feature_categories['plant_metrics']
        },
        "Microbiota Shannon (Leaf) with Plant Metrics": {
            "target": "Shannon_microbiota_leaf",
            "features": feature_categories['environmental'] + feature_categories['plant_metrics']
        },

        # 3. Pathobiota Richness
        "Pathobiota Richness (Root) with Plant OTUs": {
            "target": "richness_pathobiota_root",
            "features": feature_categories['environmental'] + feature_categories['plant_otus']
        },
        "Pathobiota Richness (Leaf) with Plant OTUs": {
            "target": "richness_pathobiota_leaf",
            "features": feature_categories['environmental'] + feature_categories['plant_otus']
        },
        "Pathobiota Richness (Root) with Plant Metrics": {
            "target": "richness_pathobiota_root",
            "features": feature_categories['environmental'] + feature_categories['plant_metrics']
        },
        "Pathobiota Richness (Leaf) with Plant Metrics": {
            "target": "richness_pathobiota_leaf",
            "features": feature_categories['environmental'] + feature_categories['plant_metrics']
        },

        # 4. Pathobiota Shannon
        "Pathobiota Shannon (Root) with Plant OTUs": {
            "target": "Shannon_pathobiota_root",
            "features": feature_categories['environmental'] + feature_categories['plant_otus']
        },
        "Pathobiota Shannon (Leaf) with Plant OTUs": {
            "target": "Shannon_pathobiota_leaf",
            "features": feature_categories['environmental'] + feature_categories['plant_otus']
        },
        "Pathobiota Shannon (Root) with Plant Metrics": {
            "target": "Shannon_pathobiota_root",
            "features": feature_categories['environmental'] + feature_categories['plant_metrics']
        },
        "Pathobiota Shannon (Leaf) with Plant Metrics": {
            "target": "Shannon_pathobiota_leaf",
            "features": feature_categories['environmental'] + feature_categories['plant_metrics']
        },
    }

    return tasks

import numpy as np

def generate_anchors(scales, ratios):
    """Generates anchor boxes for given scales and aspect ratios."""
    anchors = []
    for scale in scales:
        for ratio in ratios:
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchors.append((width, height))
    return np.array(anchors)

# Example: Scales and ratios
scales = [0.1, 0.2, 0.4]
ratios = [0.5, 1, 2]
anchors = generate_anchors(scales, ratios)
print("Anchor Boxes:", anchors)
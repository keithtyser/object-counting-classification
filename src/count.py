# src/count.py

def count_objects(classified_objects):
    """
    Count the number of each type of object in the classified objects.

    Parameters:
    - classified_objects: List of classified objects.

    Returns:
    - counts: Dictionary with counts of each object type.
    """
    counts = {'person': 0, 'bicycle': 0, 'car': 0}

    for obj in classified_objects:
        label = obj[0]
        if label in counts:
            counts[label] += 1

    return counts

from . import plot

from .plot import tshow, plot_bboxes, plot_keypoints


def _make_hand_graph():
    graph = []
    graph = graph + [(k, k + 1) for k in range(4)]  # thumb
    graph = graph + [(0, 5)] + [(k, k + 1) for k in range(5, 8)]  # 2
    graph = graph + [(0, 9)] + [(k, k + 1) for k in range(9, 12)]  # 3
    graph = graph + [(0, 13)] + [(k, k + 1) for k in range(13, 16)]  # 4
    graph = graph + [(0, 17)] + [(k, k + 1) for k in range(17, 20)]  # 5

    return graph


hand_graph = _make_hand_graph()

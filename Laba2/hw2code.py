import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    sorted_indices = np.argsort(feature_vector)
    feature_sorted = feature_vector[sorted_indices]
    target_sorted = target_vector[sorted_indices]

    unique_values = np.unique(feature_sorted)
    if len(unique_values) == 1:
        return np.array([]), np.array([]), None, -np.inf

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2

    best_gini = -np.inf
    best_threshold = None
    ginis = []

    for threshold in thresholds:
        left_mask = feature_sorted < threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        n_total = len(feature_sorted)

        if n_left == 0 or n_right == 0:
            ginis.append(-np.inf)
            continue

        left_targets = target_sorted[left_mask]
        p1_left = np.mean(left_targets == 1)
        H_left = 1 - p1_left ** 2 - (1 - p1_left) ** 2

        right_targets = target_sorted[right_mask]
        p1_right = np.mean(right_targets == 1)
        H_right = 1 - p1_right ** 2 - (1 - p1_right) ** 2

        gini = - (n_left / n_total) * H_left - (n_right / n_total) * H_right
        ginis.append(gini)

        if gini > best_gini:
            best_gini = gini
            best_threshold = threshold

    return np.array(thresholds), np.array(ginis), best_threshold, best_gini


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if any(ft not in ["real", "categorical"] for ft in feature_types):
            raise ValueError("Unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if (np.all(sub_y == sub_y[0]) or
                (self._max_depth is not None and depth >= self._max_depth) or
                len(sub_y) < self._min_samples_split):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            else:
                counts = Counter(sub_X[:, feature])
                class_counts = Counter(sub_X[sub_y == 1, feature])
                ratios = {cat: class_counts.get(cat, 0) / count for cat, count in counts.items()}
                sorted_cats = sorted(ratios.keys(), key=lambda x: ratios[x])
                cat_map = {cat: i for i, cat in enumerate(sorted_cats)}
                feature_vector = np.array([cat_map[x] for x in sub_X[:, feature]])

            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini == -np.inf or threshold is None:
                continue

            if gini_best is None or gini > gini_best:
                current_split = feature_vector < threshold

                if (np.sum(current_split) >= self._min_samples_leaf and
                        np.sum(~current_split) >= self._min_samples_leaf):

                    feature_best = feature
                    gini_best = gini
                    split = current_split

                    if feature_type == "real":
                        threshold_best = threshold
                    else:
                        threshold_best = [cat for cat in cat_map if cat_map[cat] < threshold]

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = threshold_best

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]

        if self._feature_types[feature_idx] == "real":
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])
        else:
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])
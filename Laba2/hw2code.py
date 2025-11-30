import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def find_best_split(feature_vector, target_vector):
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    unique_values = np.unique(sorted_features)

    if len(unique_values) == 1:
        return np.array([]), np.array([]), None, -np.inf

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2

    ginis = []

    for threshold in thresholds:
        left_mask = sorted_features < threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        n_total = len(sorted_features)

        if n_left == 0 or n_right == 0:
            ginis.append(-np.inf)
            continue

        left_targets = sorted_targets[left_mask]
        if len(left_targets) == 0:
            H_left = 0
        else:
            p1_left = np.sum(left_targets == 1) / n_left
            p0_left = 1 - p1_left
            H_left = 1 - p1_left ** 2 - p0_left ** 2

        right_targets = sorted_targets[right_mask]
        if len(right_targets) == 0:
            H_right = 0
        else:
            p1_right = np.sum(right_targets == 1) / n_right
            p0_right = 1 - p1_right
            H_right = 1 - p1_right ** 2 - p0_right ** 2

        gini = - (n_left / n_total) * H_left - (n_right / n_total) * H_right
        ginis.append(gini)

    ginis = np.array(ginis)

    if len(ginis) > 0 and np.max(ginis) > -np.inf:
        best_idx = np.argmax(ginis)
        threshold_best = thresholds[best_idx]
        gini_best = ginis[best_idx]
    else:
        threshold_best = None
        gini_best = -np.inf

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None,
                 min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if (np.all(sub_y == sub_y[0]) or
                (self._max_depth is not None and depth >= self._max_depth) or
                (self._min_samples_split is not None and len(sub_y) < self._min_samples_split) or
                len(sub_y) == 0):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    # Избегаем деления на ноль
                    if current_click == 0:
                        ratio[key] = float('inf')
                    else:
                        ratio[key] = current_click / current_count

                sorted_categories = [x[0] for x in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError

            # Пропускаем константные признаки
            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini is None or gini == -np.inf:
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [x[0] for x in categories_map.items() if x[1] < threshold]
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # Проверка min_samples_leaf
        if (self._min_samples_leaf is not None and
                (np.sum(split) < self._min_samples_leaf or np.sum(
                    ~split) < self._min_samples_leaf)):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        else:
            feature_idx = node["feature_split"]
            if self._feature_types[feature_idx] == "real":
                if x[feature_idx] < node["threshold"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])
            else:  # categorical
                if x[feature_idx] in node["categories_split"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
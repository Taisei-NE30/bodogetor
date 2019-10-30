from flask import Flask, render_template, json, jsonify, request
import pandas as pd
from sklearn import tree
import numpy as np

app = Flask(__name__)

tree = None
structure = None
train_X = None
train_Y = None

@app.route('/', methods=['GET', 'POST'])
def question():
    if request.method == 'GET':
        tree = generate_tree()
        structure = analysis_structure(tree)
        feature_id, threshold, children_left, children_right = get_node_info(structure, node_id=0)
        feature_word = train_X.columns[feature_id]
        return_json = {
            "feature_id": feature_id,
            "threshold": threshold,
            "children_left": children_left,
            "children_right": children_right,
        }
        return jsonify(result=json.dump(return_json))

    if request.method == 'POST':






def generate_tree():
    data = pd.read_csv('../static/bodoge_list2.csv')
    data = data.drop('designer', axis=1)

    train_X = data.drop('name', axis=1)
    train_Y = data['name']

    decision_tree = tree.DecisionTreeClassifier(random_state=0)
    decision_tree = decision_tree.fit(train_X, train_Y)
    return decision_tree


def analysis_structure(decision_tree):
    n_nodes = decision_tree.tree_.node_count
    children_left = decision_tree.tree_.children_left
    children_right = decision_tree.tree_.children_right
    feature = decision_tree.tree_.feature
    threshold = decision_tree.tree_.threshold
    values = decision_tree.tree_.value
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    stack = [(0, -1)]
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
        return n_nodes, children_left, children_right, feature, threshold, values, node_depth, is_leaves


def get_node_info(n_nodes, children_left, children_right, feature, threshold, values, is_leaves, node_id):
    name_id = 0
    if is_leaves[node_id]:
        for value in values[node_id]:
            if value > 0:
                return name_id
            name_id += 1

    else:
        return feature[node_id], threshold[node_id], children_left[n_nodes], children_right[n_nodes]

    def generate_words(feature_word, threshold_value, ):



if __name__ == '__main__':
    app.run()

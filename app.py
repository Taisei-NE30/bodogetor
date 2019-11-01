from flask import Flask, render_template, json, jsonify, request
from flask_cors import CORS
import pandas as pd
from sklearn import tree
import numpy as np

app = Flask(__name__)
CORS(app)


class LazyOnceEvalTree:
    def __init__(self, fn):
        self._fn = fn
        self.proxy = None

    def __getattr__(self, item):
        if self.proxy is None:
            self.proxy = self._fn()
        return getattr(self.proxy, item)


def get_tree():
    return generate_tree()

def generate_tree():
    data = pd.read_csv('./static/bodhoge_list2.csv')
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


def generate_words(feature_word, threshold_value):
    txt = ''
    if feature_word == 'minPlayer':
        txt += '最低{}人以上でプレイできる？'.format(int(threshold_value))
    if feature_word == 'maxPlayer':
        txt += '最高{}人以上でプレイできる？'.format(int(threshold_value))
    if feature_word == 'minTime':
        txt += '最低プレイ時間は表記{}以上？'.format(int(threshold_value))
    if feature_word == 'maxTime':
        txt += '最高プレイ時間は表記{}以上？'.format(int(threshold_value))
    if feature_word == 'age':
        txt += '対象年齢は{}以上？'.format(int(threshold_value))
    if feature_word == 'year':
        txt += '発売年は{}より後？'.format(int(threshold_value))
    if feature_word == 'auction':
        txt += 'メカニズムに競りが含まれる？'
    if feature_word == 'diceRoll':
        txt += 'ダイスを振る？'
    if feature_word == 'tilePlacement':
        txt += 'メカニズムにタイルプレイスメントが含まれる？'
    if feature_word == 'bluff':
        txt += 'ブラフ要素がある？'
    if feature_word == 'areaMajority':
        txt += 'メカニズムにエリアマジョリティが含まれる？'
    if feature_word == 'hiddenRoles':
        txt += 'メカニズムに正体隠匿が含まれる？'
    if feature_word == 'cooperative':
        txt += '協力要素がある？'
    if feature_word == 'workerPlacement':
        txt += 'メカニズムにワーカープレイスメントが含まれる？'
    if feature_word == 'balance':
        txt += 'バランスゲーム？'
    if feature_word == 'draft':
        txt += 'メカニズムにドラフトが含まれる？'
    if feature_word == 'network':
        txt += 'メカニズムにネットワーク構築が含まれる？'
    if feature_word == 'stock':
        txt += '株要素がある？'
    if feature_word == 'trickTaking':
        txt += 'トリックテイキングゲーム？'
    if feature_word == 'burst':
        txt += 'バースト要素がある？'
    if feature_word == 'setCollection':
        txt += 'セットコレクション要素がある？'
    if feature_word == 'handManagement':
        txt += 'メカニズムにハンドマネージメントが含まれる？'
    if feature_word == 'deckBuilding':
        txt += 'メカニズムにデッキビルディングが含まれる？'
    if feature_word == 'batting':
        txt += 'バッティング要素がある？'
    if feature_word == 'negotiation':
        txt += '交渉ができる？'
    if feature_word == 'team':
        txt += 'チームで戦う？'
    if feature_word == 'actionPoint':
        txt += 'メカニズムにアクションポイントが含まれる？'
    if feature_word == 'variablePhaseOrder':
        txt += 'ヴァリアブルフェイズオーダー制？'
    if feature_word == 'actionPlot':
        txt += 'メカニズムにアクションプロットが含まれる？'
    if feature_word == 'realTime':
        txt += 'リアルタイム性がある？'
    if feature_word == 'memory':
        txt += '記憶力が必要なゲーム？'
    if feature_word == 'reasoning':
        txt += '推理要素がある'
    if feature_word == 'word':
        txt += 'ワードゲーム'
    if feature_word == 'action':
        txt += 'アクション要素がある？'
    if feature_word == 'storyMaking':
        txt += 'ストーリーテリング要素がある？'
    if feature_word == 'variablePlayerPower':
        txt += '個人能力がある？'
    if feature_word == 'drawing':
        txt += '何か描く？'
    if feature_word == 'legacy':
        txt += 'レガシーゲーム？'
    if feature_word == 'escapeRoom':
        txt += '脱出ゲーム？'
    if feature_word == 'civilization':
        txt += '文明を発展させていく？'
    if feature_word == 'fantasy':
        txt += '世界観がファンタジー？'
    if feature_word == 'cthulhu':
        txt += 'クトゥルフ神話がテーマ？'
    if feature_word == 'space':
        txt += '宇宙に関係している？'
    if feature_word == 'sf':
        txt += '世界観がSF？'
    if feature_word == 'war':
        txt += '戦争がテーマ？'
    if feature_word == 'exploring':
        txt += 'どこかを探検する？'
    if feature_word == 'building':
        txt += '何かを建設する？'
    if feature_word == 'territory':
        txt += '陣地を拡大させていく？'
    if feature_word == 'animal':
        txt += '動物(人間以外)に関係がある？'
    if feature_word == 'detective':
        txt += '探偵と関係がある？'
    if feature_word == 'mafia':
        txt += 'マフィアと関係がある？'
    if feature_word == 'spy':
        txt += 'スパイと関係がある？'
    if feature_word == 'zombie':
        txt += 'ゾンビと関係がある？'
    if feature_word == 'japan':
        txt += '日本と関係がある？'
    if feature_word == 'pirate':
        txt += '海賊と関係がある？'
    if feature_word == 'farm':
        txt += '農場と関係がある？'
    if feature_word == 'music':
        txt += '音楽と関係がある？'
    if feature_word == 'sports':
        txt += 'スポーツと関係がある？'
    if feature_word == 'train':
        txt += '列車と関係がある？'
    if feature_word == 'nonTheme':
        txt += 'テーマがない？'

    return txt


tree = get_tree()
train_X = None
train_Y = None


@app.route('/')
def question_root():
    n_nodes, children_left, children_right, feature, threshold, values, node_depth, is_leaves = analysis_structure(tree)
    feature_id, threshold, children_left, children_right = get_node_info(
        n_nodes,
        children_left,
        children_right,
        feature,
        threshold,
        values,
        is_leaves,
        0
    )
    feature_word = train_X.columns[feature_id]
    return_json = {
        'txt': generate_words(feature_word, threshold),
        'children_left': children_left,
        'children_right': children_right,
        'is_leave': False
    }
    return render_template(
        'index.html',
        txt=generate_words(feature_word, threshold),
        children_left=children_left,
        children_right=children_right
    )


@app.route('/<id>')
def question(id=None):
    if id == 'yes':
        return render_template(
            'index.html',
            txt='おみとおしです',
            end=True
        )
    if id == 'no':
        return render_template(
            'index.html',
            txt='ざんねん……',
            end=True
        )

    n_nodes, children_left, children_right, feature, threshold, values, node_depth, is_leaves = analysis_structure(tree)
    feature_id, threshold, children_left, children_right = get_node_info(
        n_nodes,
        children_left,
        children_right,
        feature,
        threshold,
        values,
        is_leaves,
        int(id)
    )
    if threshold is None:
        return render_template(
            'index.html',
            txt=train_Y[feature_id],
            children_left="yes",
            children_right="no",
            is_leaves=True
        )
    else:
        feature_word = train_X.columns[feature_id]
        return render_template(
            'index.html',
            txt=generate_words(feature_word, threshold),
            children_left=children_left,
            children_right=children_right
        )


if __name__ == '__main__':
    app.run(debug=True)

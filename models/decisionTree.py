import pandas as pd
from sklearn import tree
import pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO

data = pd.read_csv('../static/bodoge_list2.csv')
data = data.drop('designer', axis=1)

train_X = data.drop('name', axis=1)
train_Y = data['name']

decision_tree = tree.DecisionTreeClassifier(random_state=0)
decision_tree = decision_tree.fit(train_X, train_Y)

dot_data = StringIO()
tree.export_graphviz(decision_tree, out_file=dot_data,feature_names=train_X.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("../static/graph.pdf")
Image(graph.create_png())



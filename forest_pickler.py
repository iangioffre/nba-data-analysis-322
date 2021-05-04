import pickle
import os
from mysklearn.myclassifiers import MyRandomForestClassifier
from mysklearn.mypytable import MyPyTable
import mysklearn.myutils as myutils

fname = os.path.join("input_data", "joined_data.csv")
table = MyPyTable().load_from_file(fname)
# year_start, height, weight, position -> season17_18
year_start_col = table.get_column("year_start", True)
height_col = table.get_column("height", True)
weight_col = table.get_discretized_column(myutils.transform_player_weight, "weight")
position_col = table.get_column("position", True)
salary_col = table.get_discretized_column(myutils.transform_salary, "season17_18")
table = [[year_start_col[i], height_col[i], weight_col[i], position_col[i], salary_col[i]] for i in range(len(salary_col))]
X = [row[:-1] for row in table]
y = [row[-1] for row in table]

header = ["year_start", "height", "weight", "position"]
rf_classifier = MyRandomForestClassifier()
rf_classifier.fit(X, y, N=80, M=7, F=3)

packaged_object = [header, rf_classifier.trees]
# pickle packaged_object
outfile = open("forest.p", "wb")
pickle.dump(packaged_object, outfile)
outfile.close()


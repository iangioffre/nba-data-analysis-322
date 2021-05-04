import os

import mysklearn.myevaluation as myevaluation
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

myevaluation.tune_parameters([50, 201], [3, 20], [2, 5], table)
import mysklearn.myutils as myutils
import copy
import csv 
import statistics
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        N = len(self.data)
        M = len(self.column_names)

        return N, M

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_index = self.column_names.index(col_identifier)
        col = []

        for row in self.data:
            if (row[col_index] != 'NA' and row[col_index] != 'N/A' or include_missing_values):
                col.append(row[col_index])

        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for index, value in enumerate(row):
                try:
                    numeric_value = float(value)
                    row[index] = numeric_value
                except ValueError:
                    pass # cannot be changed to numeric value

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        for row_to_drop in rows_to_drop:
            for row in self.data:
                if row_to_drop == row:
                    self.data.remove(row)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            first_line = True
            for row in csvreader:
                if first_line:
                    for col in row:
                        self.column_names.append(col)
                    first_line = False
                else:
                    self.data.append(row)

        self.convert_to_numeric()

        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(self.column_names)
            for row in self.data:
                csvwriter.writerow(row)

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        duplicate_rows = []
        key_column_data = []
        key_data = [] # table only populated with key columns
        for col_name in key_column_names:
            key_column_data.append(self.get_column(col_name, False))
        for i in range(len(self.data)):
            key_data.append(list(col[i] for col in key_column_data))

        for i, row1 in enumerate(key_data):
            for j, row2 in enumerate(key_data):
                if i != j and row1 == row2 and self.data[i] not in duplicate_rows:
                    duplicate_rows.append(self.data[j])

        return duplicate_rows

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        for row in self.data:
            for value in row:
                if value == 'NA' or value == 'N/A':
                    self.data.remove(row)
                    break

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        self.convert_to_numeric()
        total = 0
        count = 0

        try:
            for row in self.data:
                if row[col_index] != 'NA':
                    total += row[col_index]
                    count += 1
                
            average = total / count
            for row in self.data:
                if row[col_index] == 'NA':
                    row[col_index] = average
        except TypeError:
            pass

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        summary_table = []

        for col_name in col_names:
            new_row = []

            # collect data from the current column
            column_data = self.get_column(col_name, False)

            if len(column_data) == 0:
                continue

            # calculate summary statistics for the current column
            new_row.append(col_name)
            col_min = min(column_data)
            col_max = max(column_data)
            new_row.append(col_min)
            new_row.append(col_max)
            new_row.append((col_max + col_min) / 2)
            new_row.append(sum(column_data) / len(column_data))
            new_row.append(statistics.median(column_data))
            
            summary_table.append(new_row)

        return MyPyTable(col_names, summary_table)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        joined_table = []
        joined_header = copy.deepcopy(self.column_names)

        for col in other_table.column_names:
            if col not in joined_header:
                joined_header.append(col)
        
        key_column_data = []
        other_key_column_data = []
        key_data = [] # table only populated with key columns
        other_key_data = [] # table only populated with key columns
        for col_name in key_column_names:
            key_column_data.append(self.get_column(col_name, False))
            other_key_column_data.append(other_table.get_column(col_name, False))
        for i in range(len(self.data)):
            key_data.append(list(col[i] for col in key_column_data))
            other_key_data.append(list(col[i] for col in other_key_column_data))

        for i, row1 in enumerate(key_data):
            for j, row2 in enumerate(other_key_data):
                if row1 == row2:
                    new_row = copy.deepcopy(self.data[i])
                    for other_col_index, value in enumerate(other_table.data[j]):
                        if other_table.column_names[other_col_index] not in self.column_names:
                            new_row.append(value)
                    joined_table.append(new_row)

        return MyPyTable(joined_header, joined_table)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        joined_table = copy.deepcopy(self.data)
        joined_header = copy.deepcopy(self.column_names)
        
        

        return MyPyTable(joined_header, joined_table)
    
    def get_discretized_column(self, transformation, column_name):
        col_index = self.column_names.index(column_name)
        col = []
        
        for i, row in enumerate(self.data):
            if row != "NA" and row != "N/A":
                col.append(transformation(row[col_index]))
                
        return col

def main():
    pass

if __name__ == "__main__":
    main()
from mysklearn.myclassifiers import MyRandomForestClassifier
from mysklearn.myevaluation import train_test_split

def test_random_forest_classifier_fit():
    # interview dataset
    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]

    # test the fit method
    X = [row[:-1] for row in interview_table]
    y = [row[-1] for row in interview_table]
    X_remainder, X_test, y_remainder, y_test = train_test_split(X, y)
    rf_classifier_7 = MyRandomForestClassifier()
    rf_classifier_7.fit(X_remainder, y_remainder)
    rf_classifier_10 = MyRandomForestClassifier()
    rf_classifier_10.fit(X_remainder, y_remainder, M=10)

    assert len(rf_classifier_7.trees) == 7 # default M = 7
    assert len(rf_classifier_10.trees) == 10

    # bramer degrees dataset
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]

    # test the fit method
    X = [row[:-1] for row in degrees_table]
    y = [row[-1] for row in degrees_table]
    X_remainder, X_test, y_remainder, y_test = train_test_split(X, y)
    rf_classifier_7 = MyRandomForestClassifier()
    rf_classifier_7.fit(X_remainder, y_remainder)
    rf_classifier_15 = MyRandomForestClassifier()
    rf_classifier_15.fit(X_remainder, y_remainder, M=15, N=50, F=3)

    assert len(rf_classifier_7.trees) == 7 # default M = 7
    assert len(rf_classifier_15.trees) == 15

def test_random_forest_classifier_predict():
    # interview dataset
    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]

    # test the predict
    X = [row[:-1] for row in interview_table]
    y = [row[-1] for row in interview_table]
    X_remainder, X_test, y_remainder, y_test = train_test_split(X, y)
    rf_classifier_7 = MyRandomForestClassifier()
    rf_classifier_7.fit(X_remainder, y_remainder)
    rf_classifier_10 = MyRandomForestClassifier()
    rf_classifier_10.fit(X_remainder, y_remainder, M=10)
    rf_classifier_tuned = MyRandomForestClassifier()
    rf_classifier_tuned.fit(X_remainder, y_remainder, N=96, M=16, F=4)
    y_predicted_7 = rf_classifier_7.predict(X_test)
    y_predicted_10 = rf_classifier_10.predict(X_test)
    y_predicted_tuned = rf_classifier_tuned.predict(X_test)
    assert len(y_predicted_7) == len(y_test)
    assert len(y_predicted_10) == len(y_test)
    assert len(y_predicted_tuned) == len(y_test)

    # bramer degrees dataset
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]

    # test the predict
    X = [row[:-1] for row in degrees_table]
    y = [row[-1] for row in degrees_table]
    X_remainder, X_test, y_remainder, y_test = train_test_split(X, y)
    rf_classifier_7 = MyRandomForestClassifier()
    rf_classifier_7.fit(X_remainder, y_remainder)
    rf_classifier_15 = MyRandomForestClassifier()
    rf_classifier_15.fit(X_remainder, y_remainder, M=15, N=50, F=3)
    rf_classifier_tuned = MyRandomForestClassifier()
    rf_classifier_tuned.fit(X_remainder, y_remainder, N=96, M=16, F=4)
    y_predicted_7 = rf_classifier_7.predict(X_test)
    y_predicted_15 = rf_classifier_15.predict(X_test)
    y_predicted_tuned = rf_classifier_tuned.predict(X_test)
    assert len(y_predicted_7) == len(y_test)
    assert len(y_predicted_15) == len(y_test)
    assert len(y_predicted_tuned) == len(y_test)
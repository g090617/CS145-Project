from sklearn.svm import SVC
from sklearn import svm
import csv
import os
import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report

class svm_model():
    def __init__(self):
        self.train_x = []
        self.train_y = []

    def train(self):
        kf = KFold(n_splits=5)

        # x_train, x_test, y_train, y_test = train_test_split(self.train_x, self.train_y, test_size=0.9, random_state=0)
        # svr_rbf = SVC(kernel='rbf', C=1e3, gamma=0.1)
        # svr_lin = SVC(kernel='linear', C=1e3)
        # svr_poly = SVC(kernel='poly', C=1e3, degree=2)
        #clf = svm.SVC()
        clf = svm.SVC(kernel='linear', C=1.0)
        list_pred_y = []
        list_truth_y = []
        for train_indices, test_indices in kf.split(self.train_x):
            print("test indices: ", test_indices)
            clf.fit(self.train_x[train_indices], self.train_y[train_indices])
            predict_y = clf.predict(self.train_x[test_indices])
            print("predict_y: ", predict_y)
            list_pred_y += list(predict_y)

            # list_truth_y += list(self.train_y[test_indices])
            for i in self.train_y[test_indices]:
                temp = numpy.matrix(i)
                temp = temp.tolist()
                list_truth_y.append(temp[0][0])

            # list_pred_y.append(predict_y)
            # list_truth_y.append(self.train_y[test_indices])
            # print(list_truth_y)
            # print("mlp score ", mlp.score(self.train_x[test_indices], self.train_y[test_indices]))

        print("list_pred_y: ", list_pred_y)
        print("list_truth_y: ", list_truth_y)
        confusion = confusion_matrix(list_pred_y, list_truth_y)
        print(confusion)
        print(classification_report(list_pred_y, list_truth_y))
        average_precision = average_precision_score(list_pred_y, list_truth_y)

        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))

        # mlp.fit(x_train, y_train)
        # scores = cross_val_score(mlp, self.train_x, self.train_y, cv=5)
        # print("Training set score: %f" % mlp.score( x_test, y_test))
        # print("Training set score: ", scores)

    def read_one_input_file(self, file_name):
        table = csv.reader(open(file_name, newline=''), delimiter=',')
        for row in table:
            float_row = [float(i) for i in row]
            # print(float_row)
            self.train_x.append(float_row)

        self.train_x = numpy.matrix(self.train_x)
        print(self.train_x.shape)

    def read_one_truth_file(self, file_name):
        table = csv.reader(open(file_name, newline=''), delimiter=',')
        for row in table:

            if row == ['1'] or row == ['-1'] or row == ['0']:
                int_row = [int(i) for i in row]
                # print(int_row)
                self.train_y.append(int_row)
            else:
                print(row[0][-1])
                self.train_y.append([int(row[0][-1])])
        self.train_y = numpy.matrix(self.train_y)
        # self.train_y = self.train_y.T
        print(self.train_y.shape)


if __name__ == '__main__':
    temp = svm_model()
    # temp.train()
    temp.read_one_input_file("input/score.csv")
    temp.read_one_truth_file("truth/ground_truth.csv")
    temp.train()


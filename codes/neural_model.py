from sklearn.neural_network import MLPClassifier
import csv
import os
import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

class neural_model():
    def __init__(self):
        self.train_x = []
        self.train_y = []

    def train(self):
        kf = KFold(n_splits=5)
        # x_train, x_test, y_train, y_test = train_test_split(self.train_x, self.train_y, test_size=0.9, random_state=0)
        mlp = MLPClassifier(hidden_layer_sizes=(10,2), alpha=1e-3,
                            solver='sgd', random_state=42)

        list_pred_y = []
        list_pred_y_roc = []
        list_truth_y = []
        for train_indices, test_indices in kf.split(self.train_x):
            # print("test indices: ", test_indices)
            mlp.fit(self.train_x[train_indices], self.train_y[train_indices])
            predict_y = mlp.predict(self.train_x[test_indices])
            predict_y_roc = mlp.predict_proba(self.train_x[test_indices])[:, 1]
            # print("predict_y: ", predict_y)
            list_pred_y += list(predict_y)
            list_pred_y_roc += list(predict_y_roc)
            # print("list(predict_y_roc): ", predict_y_roc)

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
        print(classification_report(list_pred_y, list_truth_y))
        average_precision = average_precision_score(list_pred_y, list_truth_y)

        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))
        # print("list_pred_y_roc: ", list_pred_y_roc)

        # mlp.fit(x_train, y_train)
        # scores = cross_val_score(mlp, self.train_x, self.train_y, cv=5)
        # print("Training set score: %f" % mlp.score( x_test, y_test))
        # print("Training set score: ", scores)

        logit_roc_auc = roc_auc_score(list_truth_y, list_pred_y)
        fpr, tpr, thresholds = roc_curve(list_truth_y, list_pred_y_roc)
        plt.figure()
        plt.plot(fpr, tpr, label='Neural Network (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        plt.show()

    def read_one_input_file(self, file_name):
        table = csv.reader(open(file_name, newline=''), delimiter=',')
        for row in table:
            float_row = [float(i) for i in row]
            # print(float_row)
            self.train_x.append(float_row)

        self.train_x = numpy.matrix(self.train_x)
        # print(self.train_x)

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
    temp = neural_model()
    # temp.train()
    temp.read_one_input_file("data/input/score.csv")
    temp.read_one_truth_file("data/truth/ground_truth.csv")
    temp.train()






import numpy
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
from sklearn.metrics import average_precision_score
import csv
import os



class logistic_model():
    def __init__(self):
        self.train_x = []
        self.train_y = []

    def train(self):
        kf = KFold(n_splits=5)
        #x_train, x_test, y_train, y_test = train_test_split(self.train_x, self.train_y, test_size=0.3, random_state=0)
        logreg = LogisticRegression(C=1e5)
        #X_shuf, Y_shuf = shuffle(x_train, y_train)
        # print(x_train)
        # print(y_train)
        #logreg.fit(x_train, y_train)
        list_pred_y = []
        list_truth_y = []
        for train_indices, test_indices in kf.split(self.train_x):
            print("test indices: ", test_indices)
            logreg.fit(self.train_x[train_indices], self.train_y[train_indices])
            predict_y = logreg.predict(self.train_x[test_indices])
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
        # y_pred = logreg.predict(x_test)
        #
        # print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
        # conf = confusion_matrix(y_test, y_pred)
        # print(conf)
        # print(classification_report(y_test, y_pred))
        # kfold = model_selection.KFold(n_splits=3, random_state=7)
        # modelCV = LogisticRegression(C=1e5)
        # scoring = 'accuracy'
        # results = model_selection.cross_val_score(modelCV, x_train, y_train, cv=kfold, scoring=scoring)
        # print("3-fold cross validation average accuracy: %.3f" % (results.mean()))
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
    temp = logistic_model()
    # temp.train()
    temp.read_one_input_file("input/score.csv")
    temp.read_one_truth_file("truth/ground_truth.csv")
    temp.train()

# data = pd.read_csv('tweet_score_actual.csv', header=0)
#
# x=data['X']
# x=np.reshape(x,(x.size,1))
# y=data['Y']
# y=np.reshape(y,(y.size,1))
# y=y.ravel()
#
# logreg = LogisticRegression()
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
# logreg = LogisticRegression(C=1e5)
# logreg.fit(X_train, y_train)
# # model = LogisticRegression()
# # model = model.fit(x, y)
# #
# # # check the accuracy on the training set
#
# y_pred = logreg.predict(X_test)
# print X_test
# print y_pred
#
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
# kfold = model_selection.KFold(n_splits=5, random_state=7)
# modelCV = LogisticRegression(C=1e5)
# scoring = 'accuracy'
# # results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
# # print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
# confusion_matrix = confusion_matrix(y_test, y_pred)
# print(confusion_matrix)
#
# print(classification_report(y_test, y_pred))
# # logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
# # fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
# # plt.figure()
# # plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
# # plt.plot([0, 1], [0, 1],'r--')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('Receiver operating characteristic')
# # plt.legend(loc="lower right")
# # plt.savefig('Log_ROC')
# # plt.show()
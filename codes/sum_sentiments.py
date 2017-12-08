import csv
import os


class sum_sentiment_scores():

    def read_one_file(self, file_name):
        table = csv.reader(open(file_name, newline=''), delimiter=',')
        summed_features = [0.0, 0.0, 0.0, 0.0]
        count = 0
        for row in table:
            # print(list(row))
            float_row = [float(i) for i in row]
            # print(float_row)
            summed_features[0] += float_row[0]
            summed_features[1] += float_row[1]
            summed_features[2] += float_row[2]
            summed_features[3] += float_row[3]
            count += 1
        summed_features[0] /= count
        summed_features[1] /= count
        summed_features[2] /= count
        summed_features[3] /= count
        print(summed_features)
        input_file = "input" + file_name[10: 21] + ".csv"
        print(input_file)
        with open(input_file, 'a+') as csvfile:
            fieldnames = ['neg', 'neu', 'pos', 'compound']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'neg': summed_features[0],
                             'neu': summed_features[1],
                             'pos': summed_features[2],
                             'compound': summed_features[3]})

    def read_sentiment_files(self):
        directory_in_str = "sentiments/"
        directory = os.fsencode(directory_in_str)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            print(filename)
            filename_string = directory_in_str + filename[:]
            print(filename_string)
            self.read_one_file(filename_string)


if __name__ == '__main__':
    temp = sum_sentiment_scores()
    # temp.train()
    temp.read_sentiment_files()
    # temp.read_one_file("sentiments/2017-11-21-19.csv")


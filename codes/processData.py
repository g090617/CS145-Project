import json
import csv
import os

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class Preprocessor:
    dataText = ""

    def readDataFile(self):
        directory_in_str = "data/"
        directory = os.fsencode(directory_in_str)
        score_list = [0.0, 0.0, 0.0, 0.0]
        line_count = 0
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".json"):
                print(filename)
                temp_string = filename[0:13]
                print(temp_string)
                with open(directory_in_str + filename, 'r') as f:
                    for line in f:
                        sentences = [line]
                        analyzer = SentimentIntensityAnalyzer()
                        for sentence in sentences:
                            vs = analyzer.polarity_scores(sentence)
                            score_list[0] += vs['neg']
                            score_list[1] += vs['neu']
                            score_list[2] += vs['pos']
                            score_list[3] += vs['compound']
                            line_count += 1
                continue
            else:
                continue
        score_list[0] /= line_count
        score_list[1] /= line_count
        score_list[2] /= line_count
        score_list[3] /= line_count
        print(score_list)
        process_file_name = "input/score.csv"
        with open(process_file_name, 'a+') as csvfile:
            fieldnames = ['neg', 'neu', 'pos', 'compound']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'neg': score_list[0],
                             'neu': score_list[1],
                             'pos': score_list[2],
                             'compound': score_list[3]})

        # process_file_name = "sentiments/" + temp_string + ".csv"
        # with open(process_file_name, 'w') as csvfile:
        #     fieldnames = ['neg', 'neu', 'pos', 'compound']
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     with open(directory_in_str + filename, 'r') as f:
        #         for line in f:
        #             sentences = [line]
        #             analyzer = SentimentIntensityAnalyzer()
        #             for sentence in sentences:
        #                 vs = analyzer.polarity_scores(sentence)
        #                 writer.writerow({'neg': vs['neg'],
        #                                  'neu': vs['neu'],
        #                                  'pos': vs['pos'],
        #                                  'compound': vs['compound']})
        #     continue
        # else:
        #     continue


if __name__ == '__main__':
    p = Preprocessor()
    p.readDataFile()

import pandas as pd
import numpy as np


def slim(input, output, columns):
    print("Importing Raw Data: Step (1/3)")
    raw_data = pd.read_csv(input, dtype="object")

    print("Removing Unnecessary Columns: Step (2/3)")
    slimmed_data = raw_data[columns]

    print("Writing Slimmed File: Step (3/3)")
    slimmed_data.to_csv(output, index=False)


def slim_all(inputs, outputs, columns_lst):
    total = len(inputs)
    for i in range(total):
        print("Slimming: File", "(" + str(i + 1) + "/" + str(total) + ")")
        slim(inputs[i], outputs[i], columns_lst[i])


if __name__ == '__main__':
    inputs = ["/home/shared/CMV/RawData/Comments_MetaData.csv", "/home/shared/CMV/RawData/Comments_TextData.csv",
        "/home/shared/CMV/RawData/Submissions_MetaData.csv", "/home/shared/CMV/RawData/Submissions_TextData.csv"]

    outputs = ['/home/shared/CMV/SlimmedData/Slimmed_Comments_MetaData.csv',
        '/home/shared/CMV/SlimmedData/Slimmed_Comments_TextData.csv',
        '/home/shared/CMV/SlimmedData/Slimmed_Submissions_MetaData.csv',
        '/home/shared/CMV/SlimmedData/Slimmed_Submissions_TextData.csv']

    columns_lst = [["name", "parent_id", "author", "link_id"], ["author", "id", "parent_id", "body"],
        ["url", "id", "author"], ["author", "id", "title", "selftext"]]

    slim_all(inputs, outputs, columns_lst)

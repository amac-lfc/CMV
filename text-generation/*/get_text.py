import pandas as pd
import numpy as np

path = "/home/shared/CMV/"

comments_text = pd.read_csv(path + "delta_comments.csv", dtype="object")[["id","body"]]
comments_text = comments_text.rename(columns={"id":"comment", "body":"text"})
# generate list of delta-winning comments
delta_comments_list = comments_text['comment'].to_numpy()


submissions_text = pd.read_csv(path + "Slimmed_Submissions_TextData.csv", dtype="object")[["id", "title", "selftext"]]
submissions_text = submissions_text.rename(columns={"id":"submission", "selftext":"text"})
submissions_list = submissions_text['submission'].to_numpy()


sub_to_comments = pd.read_csv(path +"Slimmed_Comments_MetaData.csv", dtype="object")[["link_id", "name"]]
sub_to_comments = sub_to_comments.rename(columns={"link_id":"submission", "name":"comment"})
# remove comments that came without names...
sub_to_comments = sub_to_comments[sub_to_comments.comment.notna()]



def getSubRow(submission):
    row = submissions_text[submissions_text.submission == submission]
    return '\n'.join(str(v) for v in row.iloc[0].to_numpy()[1:])


def getCommentRow(comment):
    row = comments_text[comments_text.comment == comment]
    return '\n'.join(str(v) for v in row.iloc[0].to_numpy()[1:])


def splitRow(row):
    return row[0].split("_")[1], row[1].split("_")[1]

# print(getSubRow("16ralh"))
# print(getCommentRow("c8r3jbd"))

max_rows = -1

done_submissions = []
count = 0
build_text = ""
s2c_np = sub_to_comments.to_numpy() # submission to comments numpy

for row in s2c_np:
    if count >= max_rows and max_rows != -1:
        break
    # print(row)

    submission, comment = splitRow(row)
    if comment in delta_comments_list and submission in submissions_list:
        if submission not in done_submissions:

            build_text += "\n\n<|newsubmission|>\n\n"

            s_row = getSubRow(submission) # row from submission text data
            build_text += s_row + "\n"

        c_row = getCommentRow(comment) # row from comment text data

        build_text += "\n\n<|newcomment|>\n\n"
        build_text += c_row + "\n"

        count += 1

# write text to file
file = open("train_text_full.txt","w")
file.write(build_text)
file.close()




""" EOF """

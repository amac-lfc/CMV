import csv

lines_to_read = 10000
comments_metadata_file = "/home/shared/CMV/Slimmed_Comments_MetaData.csv"
comment_dct = {}

print("Reading through comments metadata.")
with open(comments_metadata_file, mode='r', encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Initial columns being read: {", ".join(row)}\n')
            line_count += 1

        # if row['Delta_Awarded'] == '1':
        comment_dct[row['name']] = list(row.values())[1:]

        line_count += 1
        if (line_count >= lines_to_read) and (lines_to_read>0):
              break;



submissions_metadata_file = "/home/shared/CMV/Slimmed_Submissions_MetaData.csv"
submissions_dct = {}

line_count = -1
print("Reading through submissions metadata.")
with open(submissions_metadata_file, mode='r', encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Initial columns being read: {", ".join(row)}\n')
            line_count += 1


        submissions_dct["t3_"+row['id']] = row["author"]

        line_count += 1
        if (line_count >= lines_to_read) and (lines_to_read>0):
              break;

print(comment_dct)



nested_count_dct = {}
reply_count_dct = {}

for id in comment_dct.keys():
    start = id
    sub_author = ""
    if comment_dct[start][2] in submissions_dct.keys():
        sub_author = submissions_dct[comment_dct[start][2]]

    com_author = comment_dct[start][1]


    count = 0
    reply_count = 0
    while start != comment_dct[start][2] and comment_dct[start][0] in comment_dct.keys():
        author = comment_dct[start][1]
        if author == sub_author or author == com_author:
            reply_count += 1
        count += 1
        start = comment_dct[start][0]
    reply_count_dct[id] = reply_count
    nested_count_dct[id] = count

with open('/home/shared/CMV/reply_counts.txt', 'w') as f:
    for key in reply_count_dct.keys():
        f.write(key + " " + str(reply_count_dct[key]) + "\n")

with open('/home/shared/CMV/nested_counts.txt', 'w') as f:
    for key in nested_count_dct.keys():
        f.write(key + " " + str(nested_count_dct[key]) + "\n")

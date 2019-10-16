import csv

lines_to_read = -1
input_file = "../vivian_ta_cmv_data.csv"

id_dct = {}

print("Reading through comments metadata.")
with open(input_file, mode='r', encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Initial columns being read: {", ".join(row)}\n')
            line_count += 1

        id_dct["t1_"+row['id']] = []

        line_count += 1
        if (line_count >= lines_to_read) and (lines_to_read>0):
              break;


lines_to_read = -1
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


for id in comment_dct.keys():
    start = id
    sub_author = ""
    if comment_dct[start][2] in submissions_dct.keys():
        sub_author = submissions_dct[comment_dct[start][2]]

    com_author = comment_dct[start][1]

    count = 0
    reply_count = 0
    comments = []

    # print(comment_dct[start][2])

    while start != comment_dct[start][2] and comment_dct[start][0] in comment_dct.keys():
        author = comment_dct[start][1]
        if author == sub_author or author == com_author:
            reply_count += 1
            comments.append(start)
        count += 1
        start = comment_dct[start][0]
    if start in id_dct.keys():
        id_dct[start].extend(comments)

for key in id_dct.keys():
    id_dct[key] = len(set(id_dct[key]))


with open('../vivian_ta_cmv_data.csv', mode='r', encoding="utf-8") as csvinput:
    with open('../vivian_ta_cmv_data_output.csv', mode='w', encoding="utf-8") as csvoutput:
        writer = csv.writer(csvoutput, dialect='excel', delimiter=',')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append('reply_count')
        all.append(row)

        for row in reader:
            row.append(id_dct["t1_"+row[1]])
            print(row[-1])

            all.append(row)


        writer.writerows(all)

import csv

lines_to_read = 10000
comments_metadata_file = "/home/shared/CMV/Slimmed_Comments_MetaData.csv"
comment_dct = {}

print("Reading through input file.")
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

nested_count_dct = {}

for id in comment_dct.keys():
    start = id
    count = 0
    while start != comment_dct[start][2] and comment_dct[start][0] in comment_dct.keys():
        count += 1
        start = comment_dct[start][0]
    #if id != "" and id != "\n":
    nested_count_dct[id] = count

with open('/home/shared/CMV/reply_counts.txt', 'w') as f:
    for key in nested_count_dct.keys():
        f.write(key + " " + str(nested_count_dct[key]) + "\n")

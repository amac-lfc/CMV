import csv

good_deltas = open("/home/shared/CMV/good_deltas.txt",mode='r', encoding="utf-8").read().split("\n")[:-1]

bad_deltas = open("/home/shared/CMV/bad_deltas.txt",mode='r', encoding="utf-8").read().split("\n")[:-1]

input_file = "/home/shared/CMV/Slimmed_Comments_TextData.csv"
lines_to_read = -1

delta_comments_file = open("/home/shared/CMV/delta_comments.csv",mode='w', encoding="utf-8")
nodelta_comments_file = open("/home/shared/CMV/nodelta_comments.csv", mode='w', encoding="utf-8")

delta_writer = csv.writer(delta_comments_file, dialect='excel', delimiter=',')
nodelta_writer = csv.writer(nodelta_comments_file, dialect='excel', delimiter=',')

with open(input_file, mode='r', encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            delta_writer.writerow(['author', 'id', 'parend_id', 'body'])
            nodelta_writer.writerow(['author', 'id', 'parend_id', 'body'])

        if row['id'] in good_deltas or row['id'] in bad_deltas:
            delta_writer.writerow(list(row.values()))
        else:
            nodelta_writer.writerow(list(row.values()))

        line_count += 1
        if (line_count >= lines_to_read) and (lines_to_read>0):
            break;

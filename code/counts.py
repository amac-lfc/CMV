import csv

comments_metadata_dict = {}

input_file = "/home/shared/CMV/Slimmed_Comments_MetaData.csv"
lines_to_read = -1
with open(input_file, mode='r', encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(" ".join(row))
        new_row = list(row.values())[1:]
        new_row[0] = new_row[0].split("_")[1]
        new_row[2] = new_row[2].split("_")[1]

        id = row['name']
        if 't1_' in id or 't2_' in id:
            id = id.split('_')[1]

        comments_metadata_dict[id] = new_row
        line_count += 1
        if (line_count >= lines_to_read) and (lines_to_read>0):
            break;
print("Done with Comment MetaData")

link_author_dict = {}
input_file = "/home/shared/CMV/Slimmed_Submissions_MetaData.csv"
lines_to_read = -1
with open(input_file, mode='r', encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(" ".join(row))

        link_author_dict[row['id']] = row['author']
        line_count += 1
        if (line_count >= lines_to_read) and (lines_to_read>0):
            break;
print("Done with Submission MetaData")

lost_child_count = 0
justified_comments_awarded_deltas = []
unjustified_comments_awarded_deltas = []
input_file = "/home/shared/CMV/Slimmed_Comments_TextData.csv"
lines_to_read = -1
with open(input_file, mode='r', encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(" ".join(row))

        body = row['body']
        if ('!delta' in body) or ('Δ' in body):
            if row['id'] in comments_metadata_dict.keys():
                comment_link = comments_metadata_dict[row['id']][2] #comment link from comment MetaData
            else:
                unjustified_comments_awarded_deltas.append(row['parent_id'].split('_')[1])
                continue
            if comment_link in link_author_dict.keys():
                link_author = link_author_dict[comment_link]
            else:
                unjustified_comments_awarded_deltas.append(row['parent_id'].split('_')[1])
                continue
            if row['author'] == link_author:
                justified_comments_awarded_deltas.append(row['parent_id'].split('_')[1])
        line_count += 1
        if (line_count >= lines_to_read) and (lines_to_read>0):
            break;
print("Done Finding Deltas")
print(len(justified_comments_awarded_deltas))
print(len(unjustified_comments_awarded_deltas))

with open('/home/shared/CMV/good_deltas.txt', 'w') as f:
    for item in justified_comments_awarded_deltas:
        f.write("%s\n" % item)
with open('/home/shared/CMV/bad_deltas.txt', 'w') as f:
    for item in unjustified_comments_awarded_deltas:
        f.write("%s\n" % item)

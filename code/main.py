from lib import *
import csv

write_file_name = "OutputTest.csv"
file = open(write_file_name, mode="w",encoding="utf-8", newline='')
writer = csv.writer(file, dialect='excel', delimiter=',')

#first writer header row
writer.writerow(['subreddit', 'author', 'parent_id', 'id', 'Delta_Awarded', 'body', 'body_html', 'certainty_count', 'extremity_count',
    'lexical_diversity_rounded', 'char_count_rounded', 'link_count', 'quote_count'])

print("Header row written.")

lines_to_read = 1000
with open('TextData.csv', mode='r', encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Initial columns being read: {", ".join(row)}\n')
            line_count += 1

        #calculate certainty_count, extremity_count, lexical_diversity_rounded, char_count_rounded, link_count, quote_count

        certainty_count = getCertaintyCount(row['body'])
        extremity_count = getExtremityCount(row['body'])
        lexical_diversity = getLexicalDiversity(row["body"])
        lexical_diversity_rounded = round(100 * lexical_diversity, -1)
        char_count_rounded = round(len(row["body"]), -3)
        link_count = getNumLinks(row["body_html"])
        quote_count = getNumQuotes(row["body"])

        #column order: (subreddit, author, parent_id, id, Delta_Awarded, body, body_html, certainty_count, extremity_count,
        #lexical_diversity_rounded, char_count_rounded, link_count, quote_count)
        writer.writerow([row['subreddit'], row['author'], row['parent_id'], row['id'], row['Delta_Awarded'], cleanText(row['body']), row['body_html'],
                                  certainty_count, extremity_count, lexical_diversity_rounded, char_count_rounded, link_count, quote_count])


        line_count += 1
        if line_count >= lines_to_read:
              break;
    print(f'Processed {line_count} lines.')
file.close()

from lib import *
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--o', default="OutputTest.csv", help='Number of cells in x')
parser.add_argument('--i',  default='TextData.csv', help='Number of cells in x')
parser.add_argument('--n', type=int, default=1000, help='time step')

opt = parser.parse_args()

output_file = opt.o
input_file = opt.i
lines_to_read = opt.n

def data_to_csv(output_file, input_file, lines_to_read):
    file = open(output_file, mode='w', encoding="utf-8")
    writer = csv.writer(file, dialect='excel', delimiter=',')

    #first writer header row
    writer.writerow(['subreddit', 'author', 'parent_id', 'id', 'Delta_Awarded', 'body', 'certainty_count', 'extremity_count',
        'lexical_diversity_rounded', 'char_count_rounded', 'link_count', 'quote_count'])

    with open(input_file, mode='r', encoding="utf-8") as csv_file:
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
            link_count = getNumLinks(row["body"])
            quote_count = getNumQuotes(row["body"])

            #column order: (subreddit, author, parent_id, id, Delta_Awarded, body,  certainty_count, extremity_count,
            #lexical_diversity_rounded, char_count_rounded, link_count, quote_count)
            writer.writerow([row['subreddit'], row['author'], row['parent_id'], row['id'], row['Delta_Awarded'], cleanText(row['body']),
                                      certainty_count, extremity_count, lexical_diversity_rounded, char_count_rounded, link_count, quote_count])

            line_count += 1
            if (line_count >= lines_to_read) and (lines_to_read>0):
                  break;
    print(f'Processed {line_count} lines.')
    file.close()


if __name__ == '__main__':
    data_to_csv(output_file, input_file, lines_to_read)

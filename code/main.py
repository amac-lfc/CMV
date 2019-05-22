from lib import *
import csv

common_words = open(r"../delta_words.txt",mode='r',encoding="utf-8").read().split(" ")

def data_to_csv(output_file, input_file, lines_to_read):
    file = open(output_file, mode='w', encoding="utf-8")
    writer = csv.writer(file, dialect='excel', delimiter=',')
    header = ['author', 'parent_id', 'id', 'Delta_Awarded','certainty_count', 'extremity_count',
        'lexical_diversity_rounded', 'char_count_rounded', 'link_count', 'quote_count']

    #first writer header row
    writer.writerow(header + common_words)

    with open(input_file, mode='r', encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Initial columns being read: {", ".join(row)}\n')
                line_count += 1

            #calculate certainty_count, extremity_count, lexical_diversity_rounded, char_count_rounded, link_count, quote_count
            body = row['body']
            certainty_count = getCertaintyCount(body)
            extremity_count = getExtremityCount(body)
            lexical_diversity = getLexicalDiversity(body)
            lexical_diversity_rounded = round(100 * lexical_diversity, -1)
            char_count_rounded = round(len(body), -3)
            link_count = getNumLinks(body)
            quote_count = getNumQuotes(body)

            lst = []
            for word in common_words:
                lst.append(body.count(word))


            #column order: (author, parent_id, id, Delta_Awarded, certainty_count, extremity_count,
            #lexical_diversity_rounded, char_count_rounded, link_count, quote_count)
            writer.writerow([row['author'], row['parent_id'], row['id'], row['Delta_Awarded'],
                                      certainty_count, extremity_count, lexical_diversity_rounded, char_count_rounded, link_count, quote_count] + lst)

            line_count += 1
            if (line_count >= lines_to_read) and (lines_to_read>0):
                  break;
    print(f'Processed {line_count} lines.')
    file.close()

data_to_csv("true_data.csv", "../TextData.csv", 800000)

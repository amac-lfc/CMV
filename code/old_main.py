import sys
sys.path.append('modules/')
from lib import *
import csv

common_words = open(r"/home/shared/CMV/delta_words.txt",mode='r',encoding="utf-8").read().split(" ")

lines_to_read = -1
NumWords = 200

def data_to_csv(output_file, input_file, lines_to_read, NumWords):
    file = open(output_file, mode='w', encoding="utf-8")
    writer = csv.writer(file, dialect='excel', delimiter=',')
    header = ['author', 'parend_id', 'id',  'certainty_count', 'extremity_count',
                'lexical_diversity_rounded', 'char_count_rounded', 'link_count', 'quote_count', 'questions_count',
                'bold_count', 'avgSentences_count', 'enumeration']

    #first writer header row
    writer.writerow(header + common_words[:NumWords])

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
            questions_count = getNumQuestions(body)
            bold_count = getNumBold(body)
            avgSentences_count = getNumAvgSentences(body)
            enumeration_count = getNumEnumeration(body)

            lst = []
            for word in common_words[:NumWords]:
                lst.append(body.count(word))


            #column order: (author, parent_id, id, Delta_Awarded, certainty_count, extremity_count,
            #lexical_diversity_rounded, char_count_rounded, link_count, quote_count)
            writer.writerow([row['author'], row['parend_id'], row['id'], certainty_count, extremity_count,
                            lexical_diversity_rounded, char_count_rounded, link_count, quote_count, questions_count,
                            bold_count, avgSentences_count, enumeration_count] + lst)

            line_count += 1
            if (line_count >= lines_to_read) and (lines_to_read>0):
                  break;
    print(f'Processed {line_count} lines.')
    file.close()


data_to_csv("/home/shared/CMV/Delta_Data.csv", "/home/shared/CMV/delta_comments.csv", lines_to_read, NumWords)
data_to_csv("/home/shared/CMV/NoDelta_Data.csv", "/home/shared/CMV/nodelta_comments.csv", lines_to_read, NumWords)

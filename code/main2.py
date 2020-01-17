import sys
sys.path.append('modules/')
from lib import *
import csv


# common_words = open(r"delta_words.txt",mode='r',encoding="utf-8").read().split(" ")

#grab new words

reader = csv.reader(open('../word_list.csv', 'r'))
headers = next(reader, None)
dct_keys = {0:'high arousal', 1:'low arousal', 2:'medium arousal', 3:'medium dominance', 4:'low dominance', 5:'high dominance', 6:'high valence', 7:'low valence', 8:'medium valence', 9:'examples', 10:'hedges', 11:'self references'}

dct = {}
for i in headers:
    dct[i] = []
# print(dct)

for row in reader:
   for i in range(len(row)):
       if row[i] != "":
           dct[dct_keys[i]].append(row[i])

print(dct)

def getFieldCounts(text):
    lst = [0 for i in range(len(dct_keys))]
    for i in range(len(dct.keys())):
        for word in dct[dct_keys[i]]:
            if word in text.split(" "):
                lst[i] += 1
    return(lst)


lines_to_read = -1
NumWords = 200

def data_to_csv(output_file, input_file, lines_to_read, NumWords):
    file = open(output_file, mode='w', encoding="utf-8")
    writer = csv.writer(file, dialect='excel', delimiter=',')
    dct_keys_lst = list(dct.keys())
    header = ['author', 'parend_id', 'id', 'nested_count', 'reply_count',
                'lexical_diversity_rounded', 'char_count_rounded', 'link_count', 'quote_count', 'questions_count',
                'bold_count', 'avgSentences_count', 'enumeration_count', 'excla_count'] + dct_keys_lst

    #first writer header row
    writer.writerow(header)

    print("Loading reply counts...")
    reply_counts_file = open("/home/shared/CMV/reply_counts.txt", "r").read().splitlines()

    reply_counts_dct = {}
    for line in reply_counts_file:
        lst = line.split(" ")
        if lst[0] != "":
            reply_counts_dct[lst[0].split("_")[1]] = lst[1]

    print("Loading nested counts...")
    nested_counts_file = open("/home/shared/CMV/nested_counts.txt", "r").read().splitlines()

    nested_counts_dct = {}
    for line in nested_counts_file:
        lst = line.split(" ")[:-1]
        if (len(lst) == 2):
            nested_counts_dct[lst[0].split("_")[1]] = lst[1]


    with open(input_file, mode='r', encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            #print(row["id"])

            if line_count == 0:
                print(f'Initial columns being read: {", ".join(row)}\n')
                line_count += 1

            #calculate certainty_count, extremity_count, lexical_diversity_rounded, char_count_rounded, link_count, quote_count
            body = row['body']
            clean_word_list = cleanText(body)

            field_counts = getFieldCounts(clean_word_list)

            if row["id"] in reply_counts_dct.keys():
                reply_count = reply_counts_dct[row["id"]]
            else:
                reply_count = 0

            if row["id"] in nested_counts_dct.keys():
                nested_count = nested_counts_dct[row["id"]]
            else:
                nested_count = 0

            lexical_diversity = getLexicalDiversity(body)
            lexical_diversity_rounded = round(100 * lexical_diversity, -1)
            char_count_rounded = round(len(body), -3)
            link_count = getNumLinks(body)
            quote_count = getNumQuotes(body)
            questions_count = getNumQuestions(body)
            bold_count = getNumBold(body)
            avgSentences_count = getNumAvgSentences(body)
            enumeration_count = getNumEnumeration(body)
            excla_count = getNumExcla(body)

            # lst = []
            # for word in common_words[:NumWords]:
            #     lst.append(body.count(word))


            #column order: (author, parent_id, id, Delta_Awarded, certainty_count, extremity_count,
            #lexical_diversity_rounded, char_count_rounded, link_count, quote_count)

            writer.writerow([row['author'], row['parend_id'], row['id'], nested_count, reply_count,
                            lexical_diversity_rounded, char_count_rounded, link_count, quote_count, questions_count,
                            bold_count, avgSentences_count, enumeration_count, excla_count] + field_counts)

            line_count += 1
            if (line_count >= lines_to_read) and (lines_to_read>0):
                  break;
    print(f'Processed {line_count} lines.')
    file.close()


data_to_csv("/home/shared/CMV/Delta_Data2.csv", "/home/shared/CMV/delta_comments.csv", lines_to_read, NumWords)
data_to_csv("/home/shared/CMV/NoDelta_Data2.csv", "/home/shared/CMV/nodelta_comments.csv", lines_to_read, NumWords)

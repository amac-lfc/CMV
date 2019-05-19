# Test functions
import csv
from lib import *

# Test 1
text = "i absolutely very very much love using words like certain commit and wonderful here is a link http://google.com. \n>This person said I commit wonderul things"
print(f"Number of words of certainty: {getCertaintyCount(text)}")
print(f"Number of words of extremity: {getExtremityCount(text)}")
print(f"Lexical Diversity: {getLexicalDiversity(text)}")
print(f"Number of links: {getNumLinks(text)}")
print(f"Number of quotes: {getNumQuotes(text)}")

# Test 2
text = "Hello there!\nOkay."
print(cleanText(text))


# Test 2
# Opening and Testing the CSV FIle uing CSV Module
# lines_to_read = 3
# with open(r'TextData.csv', mode='r', encoding="utf-8") as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}\n')
#             line_count += 1
#         print(f'\tAuthor:\n{row["author"]}\n\tInitial Content:\n{row["body"][0:100]}.\n')
#         line_count += 1
#         if line_count >= lines_to_read:
#               break;
#     print(f'Processed {line_count} lines.')


# Test 4
# Determining the span for Lexical Diversity and Number of Characters
# lines_to_read = 1000
#
# lexicalDiversity = {}
# numOfChars = {}
#
# with open('TextData.csv', mode='r', encoding="utf-8") as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}\n')
#             line_count += 1
#
#         # file manipulation here
#         #use row[column_name] to get data within that column
#
#         #this shows the first 100 chars of data in the comments
#         # print(f'\tAuthor:\n{row["author"]}\n\tInitial Content:\n{row["body"][0:100]}.\n')
#
#         text_length = len(row["body"])
#         text_length = round(text_length, -3)
#         if text_length not in numOfChars.keys():
#               numOfChars[text_length] = 1
#         else:
#               numOfChars[text_length] += 1
#
#         cur_lex_div = getLexicalDiversity(row["body"])
#         cur_lex_div = round(100 * cur_lex_div, -1)
#         if cur_lex_div not in lexicalDiversity.keys():
#               lexicalDiversity[cur_lex_div] = 1
#         else:
#               lexicalDiversity[cur_lex_div] += 1
#
#         #file manipulation ends here
#         line_count += 1
#         if line_count >= lines_to_read:
#               break;
#         #print(line_count)
#     print("Rounded number of characters: Occurences")
#
#     result = ""
#     for x,y in sorted(numOfChars.items()):
#         result += f"{x}: {y} | "
#     print(result[:-2],"\n")
#     #print(numOfChars,"\n")
#
#     print("Rounded lexical diversities: Occurences")
#     result = ""
#     for x,y in sorted(lexicalDiversity.items()):
#         result += f"{x}: {y} | "
#     print(result[:-2],"\n")
#
#     print(f'Processed {line_count} lines.')

#Test 4
#Getting num most common words from text
text = "i absolutely very very much love using words like certain commit and wonderful here is a link http://google.com. \n>This person said I commit wonderful things"
text = cleanText(text)
print(getCommonWords(5, text))

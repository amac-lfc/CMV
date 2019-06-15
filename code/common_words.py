""" Create program to create two files. One file containing most common words for deltas, and the other file for non-deltas """

from lib import *
import csv

lines_to_read = -1
input_file = "../delta_comments.csv"

# no_delta_words_file = open("../no_delta_words.txt", mode='w', encoding="utf-8")
delta_words_file = open("./delta_words.txt", mode='w', encoding="utf-8")

# no_delta_text = ""
delta_text = ""


print("Reading through input file.")
with open(input_file, mode='r', encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Initial columns being read: {", ".join(row)}\n')
            line_count += 1

        # if row['Delta_Awarded'] == '1':
        delta_text += row['body'].lower()

        line_count += 1
        if (line_count >= lines_to_read) and (lines_to_read>0):
              break;



print(f'Processed {line_count} lines.')

print("Cleaning Text")
delta_text = cleanText(delta_text)
#no_delta_text = cleanText(no_delta_text)

print("Organizing Most Common Words")
common_words = getCommonWords(500,delta_text)

print(common_words)
print("Writing to Text File")
delta_words_file.write(" ".join(common_words))

#no_delta_words_file.close()
delta_words_file.close()

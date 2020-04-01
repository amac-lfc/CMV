# CMV

## The Data
The DATA is from [Changed My View](https://www.reddit.com/r/changemyview/) and can be downloaded on [Data_CMV](https://www.dropbox.com/sh/e7u90jw3zwqdkit/AADqa3YjfP8a6cTxt-NyqHc9a?dl=0)

In the data folder:
04 - Comments Subset TopLevel Only
This file only includes top level comments -- i.e., comments that are in direct response to the initial submission itself
* Includes top level comments that were awarded and were not awarded a delta
4,555 top level comments were awarded deltas out of a total of 737,507 top-level comments
that's roughly 0.065% of all top level comments
This also means that roughly 2/3 of all deltas awarded result from some back-and-forth (i.e., 4816/13582)
As a side note, there were about 4.5 million comments in total. So, this dataset contains about 16% of all comments. Put another way, ~16% of all comments were top-level comments.

* This should not include any top level comments where a delta was awarded downstream
For example, if someone makes a CMV submission, and I leave a comment, then we have a few back-and-forth comments, and she eventually awards me a delta downstream, the entire chain is omitted

* Should not include any top-level comments made by the OP. So, I've seen a couple where the OP didn't assign a delta to a specific comment, but just left a top-level comment in their own thread saying something like "wow, everyone here made good points. I'm giving a delta to the whole group". Things like this are not included in this file.

* Only treats delta comments as having been awarded a delta where the submission's OP is the person who commented with a delta. So, if someone made a thread, and I left a top level comment, and Sarah came in and said "!delta" in response to my comment, that doesn't count as a delta here.

* Deltas didn't get counted if the OP assigned one to themselves. Not sure if this actually happened at all in the data, but I built the code to make sure that it didn't.

* Omitted comments where the body text was either [removed] or [deleted] (see more info below)

* Omitted comments made by the AutoModerator (note that one user has the name "AutoModerater", but they're not actually an automod)

* So, long story short, this file (the one in the "04" folder) has top-level comments that either did or did not receive a delta. Most of them don't but for those that do, I've also included the text where the OP assigned the delta to them.


## Machine Learning Algorithms
Decision Trees:
* [Machine Learning 101](https://medium.com/machine-learning-101)
  * [Chapter 3: Decision Tree Classifier — Coding](https://medium.com/machine-learning-101/chapter-3-decision-tree-classifier-coding-ae7df4284e99)
  * [Chapter 4: K Nearest Neighbors Classifier](https://medium.com/machine-learning-101/k-nearest-neighbors-classifier-1c1ff404d265)
  * [Chapter 5: Random Forest Classifier](https://medium.com/machine-learning-101/chapter-5-random-forest-classifier-56dc7425c3e1)


## Running the Program

### Quick Order to Run Code
1. Run FilerData.py
2. Run counts.py
3. Run separateData.py
4. Run common_words.py
5. Run main.py or main2.py
6. Run shufflecsv.py or shufflecsv2.py
7. Run classifier.py or aba.gb.py or xgboost_clf.py or new_words_ada_gb.py or new_words_classifier.py


### Filtering Data

The original CSV data from the subreddit came as raw data and had special characters within the document.
Since it's difficult to do NLP on text with special characters, we replace the special text with the latin
equivalent. The document is also massive so we do cut unnecessary columns from the CSV data.

If you would like to change the files that are being used, you can edit the lines below in the code.

```python
# File with extra information about the comments.
input_file = "/home/shared/CMV/01 - RawData/01 - RawData/2019-02-11 - Extracted Reddit Data - Comments - MetaData.csv"

# File with the comment data.
input_file = "/home/shared/CMV/RawData/2019-02-11 - Extracted Reddit Data - Comments - TextData.csv"

# File with information about the submissions.
input_file = "/home/shared/CMV/RawData/2019-02-11 - Extracted Reddit Data - Submissions - MetaData.csv"

# File with the submission data.
input_file = "/home/shared/CMV/RawData/2019-02-11 - Extracted Reddit Data - Submissions - TextData.csv"
```
Run in the terminal `python3 FilterData.py` to run the data filter.


### Finding the Deltas

In order to run Prediction Algorithms on the Delta-Winning comments, the Delta-Winning Comments must first be found.
The code searches all comments for either _!delta_ or _Δ_, and adds the parent of the delta comment to a list of
Delta-Winning comments. Some comments that are Delta-Winning may not be in the Meta Data, in which case we add it to
a bad_deltas category, otherwise the comments go into a good_deltas category.

If you would like to change the files that are being inputted, you can edit the lines below in the code.

```python
# This is the extra information about the comments, slimmed.
input_file = "/home/shared/CMV/Slimmed_Comments_MetaData.csv"

# This is the extra information about the submissions, slimmed.
input_file = "/home/shared/CMV/Slimmed_Submissions_MetaData.csv"

# This is the slimmed comments.
input_file = "/home/shared/CMV/Slimmed_Comments_TextData.csv"
```

To change the output files created, change the below lines in the code.

```python
# This is the file for the comments with Meta Data.
with open('/home/shared/CMV/good_deltas.txt', 'w') as f:

# This is the file for the comments without Meta Data.
with open('/home/shared/CMV/bad_deltas.txt', 'w') as f:
```
Run in the terminal `python3 counts.py` to run the data filter.


### Splitting the Deltas and Non Deltas

Once that the deltas have been found, the comments data can not split comments into delta-winning comments or
non-delta winning comments. The command goes through all of the comment data and if they are in either bad deltas or good deltas,
add them to a list of delta comments, otherwise add them in a list of non-delta comments.

To make it so you only use comments with meta data, edit the line in the code below.

```python
if row['id'] in good_deltas or row['id'] in bad_deltas:
```

If you would like to change the files that are being inputted, you can edit the lines below in the code.

```python
# This is the file for the comments with Meta Data.
good_deltas = open("/home/shared/CMV/good_deltas.txt",mode='r', encoding="utf-8").read().split("\n")[:-1]

# This is the file for the comments without Meta Data.
bad_deltas = open("/home/shared/CMV/bad_deltas.txt",mode='r', encoding="utf-8").read().split("\n")[:-1]

# This is the slimmed comments.
input_file = "/home/shared/CMV/Slimmed_Comments_TextData.csv"
```

To change the output files created, change the below lines in the code.

```python
# The delta-winning comments.
delta_comments_file = open("/home/shared/CMV/delta_comments.csv",mode='w', encoding="utf-8")

# The non-delta-winning comments.
nodelta_comments_file = open("/home/shared/CMV/nodelta_comments.csv", mode='w', encoding="utf-8")
```

Run in the terminal `python3 separateData.py` to separate the Delta and Non-Delta comments.


### Generating Most Common Words

Part of the program can keep track of the most commonly used words for both the Delta-Winning Comments.
The most commonly use words are generated by keeping track of the number
of times a certain word is found in the comments. Then the top 500 words get saved in a different file.

To change the number of words saved, input file and output file, edit the below line in the code.

```python
num_of_words = 500

input_file = "/home/shared/CMV/delta_comments.csv"

delta_words_file = open("./delta_words.txt", mode='w', encoding="utf-8")
```

Run in the terminal `python3 common_words.py` to generate the file with the most common words.


### Feature Engineering

Then the features must be generated in order to classify our data.

For these features to be generated:
* nested_count
* reply_count
* certainty_count
* extremity_count
* lexical_diversity_rounded
* char_count_rounded
* link_count
* quote_count
* questions_count
* bold_count
* avgSentences_count
* enumeration_count
* excla_count

Run the command `python3 main.py`.

For the above features and the features below to be generated:
* high arousal
* low arousal
* medium arousal
* medium dominance
* low dominance
* high dominance
* high valence
* low valence
* medium valence
* examples
* hedges
* self references

Run the command `python3 main2.py`

To change the input or output files, edit the below lines found within the code.
```python
# First file is output, second is input.

# Delta data
data_to_csv("/home/shared/CMV/Delta_Data.csv", "/home/shared/CMV/delta_comments.csv", lines_to_read, NumWords)

#Nondelta Data
data_to_csv("/home/shared/CMV/NoDelta_Data.csv", "/home/shared/CMV/nodelta_comments.csv", lines_to_read, NumWords)
```

### Selecting the Comments for Use

Since our data is uneven, we shuffle part of our major NonDelta Comments and select a fraction of them for use.

You can run `python3 shufflecsv.py` to select the data with the features from main.py
and `python3 shufflecsv2.py` for the data with the features from main2.py.


### Classifying

Once that has been all the above has been finished, you can run any of the below lines to use different classifiers
on the features and data.

* `python3 classifier.py`
* `python3 aba_gb.py`
* `python3 xgboost_clf.py`
* `python3 new_words_ada_gb.py`
* `python3 new_words_classifier.py`

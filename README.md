# CMV

##How to run code
1. Run FilerData.py
2. Run counts.py
3. Run separateData.py
4. Run common_words.py
5. Run main.py
6. Run shufflecsv.py
7. Run classifier.py



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

**Possible criteria for the algorithm:**
* length of response
* whether they provided links and how many links
* if they quoted the OP

## Machine Learning Algorithms
Decision Trees:
* [Machine Learning 101](https://medium.com/machine-learning-101)
  * [Chapter 3: Decision Tree Classifier — Coding](https://medium.com/machine-learning-101/chapter-3-decision-tree-classifier-coding-ae7df4284e99)
  * [Chapter 4: K Nearest Neighbors Classifier](https://medium.com/machine-learning-101/k-nearest-neighbors-classifier-1c1ff404d265)
  * [Chapter 5: Random Forest Classifier](https://medium.com/machine-learning-101/chapter-5-random-forest-classifier-56dc7425c3e1)


## To-Do List

1. Clean the data -> write 04 into a *csv* file with the variables defined
2. Learning Sklearn/Decision tree/Random Forest
3. Write the algorithm
4. Test the data

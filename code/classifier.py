import csv
from random import shuffle
from sklearn import tree
from sklearn.metrics import accuracy_score

def grab(value,data):
    grabbed_data = []
    for i in data:
        if i[1] == value:
            grabbed_data.append(i)
    return grabbed_data

def evenOutData(data, max_lines):
    with_delta = 0
    no_delta = 0

    for i in data:
        if i[1] == '0':
            no_delta += 1
        else:
            with_delta += 1
    num_of_each = min(with_delta, no_delta, max_lines)
    print(f'Using {2 * num_of_each} Lines from Data')

    with_delta = 0
    no_delta = 0
    new_data = []

    for i in data:
        if i[1] == '0' and no_delta < num_of_each:
            no_delta += 1
            new_data.append(i)
        elif i[1] == '1' and with_delta < num_of_each:
            with_delta += 1
            new_data.append(i)

    return new_data

def separateData(data):
    features = []
    labels = []
    for x,y in data:
        features.append(x)
        labels.append(y)
    return features, labels

def readInputFile(input_file, lines_to_read):
    data = []

    with open(input_file, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            feature = [row['certainty_count'], row['extremity_count'], row['lexical_diversity_rounded'],
                       row['char_count_rounded'], row['link_count'], row['quote_count']]
            data.append([feature,row['Delta_Awarded']])
            line_count += 1
            if (line_count >= lines_to_read) and (lines_to_read>0):
                  break;
    print(f'Processed {line_count} lines.')
    return data

print('Reading File and Creating Data')
data = readInputFile("../data.csv",1000000)

print("Randomizing and Evening Out Data")
shuffle(data)
fixed_data = evenOutData(data,5000)

print("Splitting Data into Train and Test")
train_data = fixed_data[:int(len(fixed_data) *.8)]
test_data = fixed_data[int(len(fixed_data) * .8):]
x_train, y_train = separateData(train_data)

#This makes it so the test data only contains deltas
delta_test_data = grab('1', test_data)
x_test, y_test = separateData(delta_test_data)

clf_tree = tree.DecisionTreeClassifier()
print("Training Decision Tree")
clf_tree = clf_tree.fit(x_train,y_train)


print("Checking for Accuracy")
y_predict = clf_tree.predict(x_test)
print(f"Accuracy score is: {accuracy_score(y_test, y_predict)}")


import os
import csv
import time
import pickle


SubmissionsMetaFile = '02 - Slimmed MetaData/2019-02-11 - Extracted Reddit Data - Submissions - MetaData.csv'
CommentsTextFile = '01 - RawData/2019-02-11 - Extracted Reddit Data - Comments - TextData.csv'
CommentsMetaFile = '02 - Slimmed MetaData/2019-02-11 - Extracted Reddit Data - Comments - MetaData.csv'

SubmissionIDname = 'id'
SubmissionAuthorname = 'author'

CommentIDName = 'id'
CommentParentName = 'parent_id'
CommentTextColumnName = 'body'
CommentAuthorColName = 'author'


authors_to_omit = []

Replacements = [["&nbsp;", " "], ["&iexcl;", "¡"], ["&cent;", "¢"], ["&pound;", "£"], ["&curren;", "¤"], ["&yen;", "¥"], ["&brvbar;", "¦"], ["&sect;", "§"], ["&uml;", "¨"], ["&copy;", "©"], ["&ordf;", "ª"], ["&laquo;", "«"], ["&not;", "¬"], ["&reg;", "®"], ["&macr;", "¯"], ["&deg;", "°"], ["&plusmn;", "±"], ["&sup2;", "²"], ["&sup3;", "³"], ["&acute;", "´"], ["&micro;", "µ"], ["&para;", "¶"], ["&cedil;", "¸"], ["&sup1;", "¹"], ["&ordm;", "º"], ["&raquo;", "»"], ["&frac14;", "¼"], ["&frac12;", "½"], ["&frac34;", "¾"], ["&iquest;", "¿"], ["&times;", "×"], ["&divide;", "÷"], ["&Agrave;", "À"], ["&Aacute;", "Á"], ["&Acirc;", "Â"], ["&Atilde;", "Ã"], ["&Auml;", "Ä"], ["&Aring;", "Å"], ["&AElig;", "Æ"], ["&Ccedil;", "Ç"], ["&Egrave;", "È"], ["&Eacute;", "É"], ["&Ecirc;", "Ê"], ["&Euml;", "Ë"], ["&Igrave;", "Ì"], ["&Iacute;", "Í"], ["&Icirc;", "Î"], ["&Iuml;", "Ï"], ["&ETH;", "Ð"], ["&Ntilde;", "Ñ"], ["&Ograve;", "Ò"], ["&Oacute;", "Ó"], ["&Ocirc;", "Ô"], ["&Otilde;", "Õ"], ["&Ouml;", "Ö"], ["&Oslash;", "Ø"], ["&Ugrave;", "Ù"], ["&Uacute;", "Ú"], ["&Ucirc;", "Û"], ["&Uuml;", "Ü"], ["&Yacute;", "Ý"], ["&THORN;", "Þ"], ["&szlig;", "ß"], ["&agrave;", "à"], ["&aacute;", "á"], ["&acirc;", "â"], ["&atilde;", "ã"], ["&auml;", "ä"], ["&aring;", "å"], ["&aelig;", "æ"], ["&ccedil;", "ç"], ["&egrave;", "è"], ["&eacute;", "é"], ["&ecirc;", "ê"], ["&euml;", "ë"], ["&igrave;", "ì"], ["&iacute;", "í"], ["&icirc;", "î"], ["&iuml;", "ï"], ["&eth;", "ð"], ["&ntilde;", "ñ"], ["&ograve;", "ò"], ["&oacute;", "ó"], ["&ocirc;", "ô"], ["&otilde;", "õ"], ["&ouml;", "ö"], ["&oslash;", "ø"], ["&ugrave;", "ù"], ["&uacute;", "ú"], ["&ucirc;", "û"], ["&uuml;", "ü"], ["&yacute;", "ý"], ["&thorn;", "þ"], ["&yuml;", "ÿ"], ["&amp;", "&"], ["&lt;", "<"], ["&gt;", ">"], ["&OElig;", "Œ"], ["&oelig;", "œ"], ["&Scaron;", "Š"], ["&scaron;", "š"], ["&Yuml;", "Ÿ"], ["&fnof;", "ƒ"], ["&circ;", "ˆ"], ["&tilde;", "˜"], ["&ndash;", "–"], ["&mdash;", "—"], ["&lsquo;", "‘"], ["&rsquo;", "’"], ["&sbquo;", "‚"], ["&ldquo;", "“"], ["&rdquo;", "”"], ["&bdquo;", "„"], ["&dagger;", "†"], ["&Dagger;", "‡"], ["&bull;", "•"], ["&hellip;", "…"], ["&permil;", "‰"], ["&lsaquo;", "‹"], ["&rsaquo;", "›"], ["&euro;", "€"], ["&trade;", "™"]]






#PULL OUT ALL OF OUR SUBMISSION IDs


print("Collecting all submissions IDs, linking with authors")

#list of all submission IDs
SubmissionIDset = set()
#tells us who the OP is for each thread
SubmissionKeyAuthorsValue = {}
AuthorsKeySubmissionValue = {}
NumberOfSubmissions = 0

with open(SubmissionsMetaFile, 'r', encoding='utf-8') as SubmissionsIn:

    csvreader = csv.reader(SubmissionsIn)

    header_row = csvreader.__next__()

    SubmissionColumnIndex = header_row.index(SubmissionIDname)
    SubmissionAuthorColumnIndex = header_row.index(SubmissionAuthorname)

    for row in csvreader:

        NumberOfSubmissions += 1
        SubmissionIDset.add(row[SubmissionColumnIndex])

        SubmissionKeyAuthorsValue[row[SubmissionColumnIndex]] = row[SubmissionAuthorColumnIndex]
        AuthorsKeySubmissionValue[row[SubmissionAuthorColumnIndex]] = row[SubmissionColumnIndex]

        

#When looking at Parent IDs, here is the guide:
        #When looking at Parent IDs, here is the guide:
        #When looking at Parent IDs, here is the guide:
        #When looking at Parent IDs, here is the guide:

##If it's a comment, the prefix is t1. Here's the other prefixes from the docs
##
##t1_ Comment
##t2_ Account
##t3_ Link
##t4_ Message
##t5_ Subreddit
##t6_ Award








print('Connecting comments to submissions')

CommentToLinkDict = {}

with open(CommentsMetaFile, 'r', encoding='utf-8') as CommentsIn:

    csvreader = csv.reader(CommentsIn)
    header_row = csvreader.__next__()
    CommentID = header_row.index(CommentIDName)
    LinkID = header_row.index('link_id')

    for row in csvreader:

        CommentToLinkDict[row[CommentID]] = row[LinkID].split('_')[1]













Deltas_Awarded_By = {}

print("Collecting all Delta IDs")

#Track the list of comments that have had deltas awarded by OP

DeltaAwardedID = set()
DeltaReceivedID = set()
DeltaReceivedIDTopLevel = set()
DeltaReceivedButNotTopLevel = set()

DeltaText = {}

with open(CommentsTextFile, 'r', encoding='utf-8') as CommentsIn:

    csvreader = csv.reader(CommentsIn)

    header_row = csvreader.__next__()
    CommentBodyID = header_row.index(CommentTextColumnName)
    CommentParentID = header_row.index(CommentParentName)
    CommentAuthorID = header_row.index(CommentAuthorColName)
    CommentID = header_row.index(CommentIDName)


    with open('03 - All Comments Awarding Deltas/2019-02-11 - Extracted Reddit Data - Comments - TextData.csv', 'w', encoding='utf-8', newline='') as outgoing:

        csvwriter = csv.writer(outgoing, dialect='excel')
        #csvwriter2 = csv.writer(outgoing2, dialect='excel')

        csvwriter.writerow(header_row)
        #csvwriter2.writerow(header_row)



        #find out which people gave deltas, and find out which comments received them
        for row in csvreader:

            #if ('33ltmy' in row[CommentParentID]) or ('cqm3qgs' in row[CommentParentID]):
            #    csvwriter2.writerow(row)

            if ('!delta' in row[CommentBodyID]) or ('Δ' in row[CommentBodyID]):

                row_to_write = row
                for replacement in Replacements:
                    row_to_write[CommentBodyID] = row_to_write[CommentBodyID].replace(replacement[0], replacement[1])

                csvwriter.writerow(row_to_write)
                

                #make sure that the person giving the delta is the OP
                if (CommentToLinkDict[row[CommentID]] in SubmissionKeyAuthorsValue.keys()) and (row[CommentAuthorID] == SubmissionKeyAuthorsValue[CommentToLinkDict[row[CommentID]]]):

                    DeltaAwardedID.add(row[CommentID])

                    #we SPLIT here because the actual parent id's are on the right half of the listed parent_id.
                    #if the parent id starts with t1, it's the top level comment. if it starts with t3, it's a response to a comment
                    ParentID_Data = row[CommentParentID].split('_')

                    
                    #we only capture parent_id's if a delta was given in resposne to a COMMENT (i.e., parent_id starts with t1)
                    #otherwise, we don't want to capture deltas that are, for whatever reason, given at the top level
                    if ParentID_Data[0] == 't1':
                        DeltaReceivedID.add(ParentID_Data[1])
                        Deltas_Awarded_By[ParentID_Data[1]] = row[CommentAuthorID]
                        DeltaText[ParentID_Data[1]] = row[CommentBodyID]



print('Found ' + str(len(DeltaReceivedID)) + ' comments that received deltas')




with open('DeltaAwardText.pickle', 'wb') as outgoing:
    pickle.dump(DeltaText, outgoing, protocol=pickle.HIGHEST_PROTOCOL)








#MAP OUT WHICH COMMENTS GOT DELTAS AND WERE TOP VERSUS NOT TOP LEVEL
print('Mapping out which comments got deltas and were top versus not top level')
with open(CommentsMetaFile, 'r', encoding='utf-8') as CommentsIn:

    csvreader = csv.reader(CommentsIn)
    header_row = csvreader.__next__()
    #CommentBodyID = header_row.index(CommentTextColumnName)
    CommentParentID = header_row.index(CommentParentName)
    CommentAuthorID = header_row.index(CommentAuthorColName)
    CommentID = header_row.index(CommentIDName)

    for row in csvreader:

        #if a delta was given in this thread, but
        if row[CommentID] in DeltaReceivedID:

            ParentID_Data = row[CommentParentID].split('_')

            if ParentID_Data[0] == 't3':
                DeltaReceivedIDTopLevel.add(row[CommentID])
            elif ParentID_Data[0] == 't1':
                DeltaReceivedButNotTopLevel.add(ParentID_Data[1])
                DeltaReceivedButNotTopLevel.add(row[CommentID])
                


print('Found ' + str(len(DeltaReceivedIDTopLevel)) + ' top-level comments that received deltas')
print('Found ' + str(len(DeltaReceivedButNotTopLevel)) + ' NON-top-level comments that received deltas')










#Here, we have to loop through to pull out all of those threads where
#deltas were EVENTUALLY given, but they weren't top level

CommentCheckPassNumber = 1
NewCommentsIdentified = 0


while (NewCommentsIdentified > 0) or (CommentCheckPassNumber == 1):

    print("Checking delta trees, Pass #" + str(CommentCheckPassNumber))

    NewCommentsIdentified = 0

    with open(CommentsMetaFile, 'r', encoding='utf-8') as CommentsIn:

        csvreader = csv.reader(CommentsIn)
        header_row = csvreader.__next__()
        #CommentBodyID = header_row.index(CommentTextColumnName)
        CommentParentID = header_row.index(CommentParentName)
        CommentAuthorID = header_row.index(CommentAuthorColName)
        CommentID = header_row.index(CommentIDName)

        for row in csvreader:

            ParentID_Data = row[CommentParentID].split('_')

            if (row[CommentID] in DeltaReceivedButNotTopLevel) and (ParentID_Data[1] not in DeltaReceivedButNotTopLevel):

                if ParentID_Data[0] == 't1':

                    DeltaReceivedButNotTopLevel.add(ParentID_Data[1])
                    NewCommentsIdentified += 1

    print(str(NewCommentsIdentified) + " new non-top delta comments found")
    CommentCheckPassNumber += 1


            
    







#save out all of the data collected thus far

with open('Aggregated_-_Deltas_TopLevel.txt', 'w', encoding='utf-8', newline='') as outgoing:
    for item in DeltaReceivedIDTopLevel:
          outgoing.write(item + '\r\n')
    
    
with open('Aggregated_-_Deltas_NonTopLevel.txt', 'w', encoding='utf-8', newline='') as outgoing:
    for item in DeltaReceivedButNotTopLevel:
          outgoing.write(item + '\r\n')

with open('Aggregated_-_Deltas_Given.txt', 'w', encoding='utf-8', newline='') as outgoing:
    for item in DeltaAwardedID:
          outgoing.write(item + '\r\n')
    
with open('Aggregated_-_Comment_to_Submission_Links.txt', 'w', encoding='utf-8', newline='') as outgoing:
    for key, value in CommentToLinkDict.items():
        outgoing.write(key + '\t' + value + '\r\n')

with open('Aggregated_-_Submission_to_Author_Links.txt', 'w', encoding='utf-8', newline='') as outgoing:
    for key, value in SubmissionKeyAuthorsValue.items():
        outgoing.write(key + '\t' + value + '\r\n')

with open('Aggregated_-_Who_Awarded_Deltas.txt', 'w', encoding='utf-8', newline='') as outgoing:
    for key, value in Deltas_Awarded_By.items():
        outgoing.write(key + '\t' + value + '\r\n')   







import os
import csv
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


Replacements = [["&nbsp;", " "], ["&iexcl;", "¡"], ["&cent;", "¢"], ["&pound;", "£"], ["&curren;", "¤"], ["&yen;", "¥"], ["&brvbar;", "¦"], ["&sect;", "§"], ["&uml;", "¨"], ["&copy;", "©"], ["&ordf;", "ª"], ["&laquo;", "«"], ["&not;", "¬"], ["&reg;", "®"], ["&macr;", "¯"], ["&deg;", "°"], ["&plusmn;", "±"], ["&sup2;", "²"], ["&sup3;", "³"], ["&acute;", "´"], ["&micro;", "µ"], ["&para;", "¶"], ["&cedil;", "¸"], ["&sup1;", "¹"], ["&ordm;", "º"], ["&raquo;", "»"], ["&frac14;", "¼"], ["&frac12;", "½"], ["&frac34;", "¾"], ["&iquest;", "¿"], ["&times;", "×"], ["&divide;", "÷"], ["&Agrave;", "À"], ["&Aacute;", "Á"], ["&Acirc;", "Â"], ["&Atilde;", "Ã"], ["&Auml;", "Ä"], ["&Aring;", "Å"], ["&AElig;", "Æ"], ["&Ccedil;", "Ç"], ["&Egrave;", "È"], ["&Eacute;", "É"], ["&Ecirc;", "Ê"], ["&Euml;", "Ë"], ["&Igrave;", "Ì"], ["&Iacute;", "Í"], ["&Icirc;", "Î"], ["&Iuml;", "Ï"], ["&ETH;", "Ð"], ["&Ntilde;", "Ñ"], ["&Ograve;", "Ò"], ["&Oacute;", "Ó"], ["&Ocirc;", "Ô"], ["&Otilde;", "Õ"], ["&Ouml;", "Ö"], ["&Oslash;", "Ø"], ["&Ugrave;", "Ù"], ["&Uacute;", "Ú"], ["&Ucirc;", "Û"], ["&Uuml;", "Ü"], ["&Yacute;", "Ý"], ["&THORN;", "Þ"], ["&szlig;", "ß"], ["&agrave;", "à"], ["&aacute;", "á"], ["&acirc;", "â"], ["&atilde;", "ã"], ["&auml;", "ä"], ["&aring;", "å"], ["&aelig;", "æ"], ["&ccedil;", "ç"], ["&egrave;", "è"], ["&eacute;", "é"], ["&ecirc;", "ê"], ["&euml;", "ë"], ["&igrave;", "ì"], ["&iacute;", "í"], ["&icirc;", "î"], ["&iuml;", "ï"], ["&eth;", "ð"], ["&ntilde;", "ñ"], ["&ograve;", "ò"], ["&oacute;", "ó"], ["&ocirc;", "ô"], ["&otilde;", "õ"], ["&ouml;", "ö"], ["&oslash;", "ø"], ["&ugrave;", "ù"], ["&uacute;", "ú"], ["&ucirc;", "û"], ["&uuml;", "ü"], ["&yacute;", "ý"], ["&thorn;", "þ"], ["&yuml;", "ÿ"], ["&amp;", "&"], ["&lt;", "<"], ["&gt;", ">"], ["&OElig;", "Œ"], ["&oelig;", "œ"], ["&Scaron;", "Š"], ["&scaron;", "š"], ["&Yuml;", "Ÿ"], ["&fnof;", "ƒ"], ["&circ;", "ˆ"], ["&tilde;", "˜"], ["&ndash;", "–"], ["&mdash;", "—"], ["&lsquo;", "‘"], ["&rsquo;", "’"], ["&sbquo;", "‚"], ["&ldquo;", "“"], ["&rdquo;", "”"], ["&bdquo;", "„"], ["&dagger;", "†"], ["&Dagger;", "‡"], ["&bull;", "•"], ["&hellip;", "…"], ["&permil;", "‰"], ["&lsaquo;", "‹"], ["&rsaquo;", "›"], ["&euro;", "€"], ["&trade;", "™"]]


authors_to_omit = set(['AutoModerator', ])
comments_to_omit = set(['[removed]', '[deleted]'])


print('Loading up all of the metadata for linking...')

TopLevelDeltaComments = set()
with open('Aggregated_-_Deltas_TopLevel.txt', 'r', encoding='utf-8') as incoming:
    TopLevelDeltaComments = set(incoming.read().splitlines())

LowLevelDeltaComments = set()
with open('Aggregated_-_Deltas_NonTopLevel.txt', 'r', encoding='utf-8') as incoming:
    LowLevelDeltaComments = set(incoming.read().splitlines())

WhoAwardedDeltas = {}
with open('Aggregated_-_Who_Awarded_Deltas.txt', 'r', encoding='utf-8') as incoming:
     AwardList = incoming.read().splitlines()

     for line in AwardList:
         line_split = line.split('\t')

         WhoAwardedDeltas[line_split[0]] = line_split[1]

CommentsToSubmissionLinks = {}
with open('Aggregated_-_Comment_to_Submission_Links.txt', 'r', encoding='utf-8') as incoming:
     ContentList = incoming.read().splitlines()

     for line in ContentList:
         line_split = line.split('\t')

         CommentsToSubmissionLinks[line_split[0]] = line_split[1]


SubmissionsToAuthorsLinks = {}
with open('Aggregated_-_Submission_to_Author_Links.txt', 'r', encoding='utf-8') as incoming:
     ContentList = incoming.read().splitlines()

     for line in ContentList:
         line_split = line.split('\t')

         SubmissionsToAuthorsLinks[line_split[0]] = line_split[1]


CommentsWhereDeltasGiven = set()
with open('Aggregated_-_Deltas_Given.txt', 'r', encoding='utf-8') as incoming:
    CommentsWhereDeltasGiven = set(incoming.read().splitlines())


DeltaAwardTextDict = {}
with open('DeltaAwardText.pickle', 'rb') as incoming:
    DeltaAwardTextDict = pickle.load(incoming)










print('Subsetting comment data...')

with open(CommentsTextFile, 'r', encoding='utf-8') as CommentTextIncoming:

    csvreader = csv.reader(CommentTextIncoming)

    with open('04 - Comments Subset TopLevel Only/2019-02-15 - Extracted Reddit Data - Comments - TextData.csv', 'w', encoding='utf-8', newline='') as CommentTextOutgoing:





        csvwriter = csv.writer(CommentTextOutgoing, dialect='excel')
        
        header_row = csvreader.__next__()
        header_row.append('Delta_Awarded')
        header_row.append('Delta_Awarder')
        header_row.append('Delta_AwardText')
        header_row.append('Submission_Author')

        CommentBodyID = header_row.index(CommentTextColumnName)
        CommentParentID = header_row.index(CommentParentName)
        CommentAuthorID = header_row.index(CommentAuthorColName)
        CommentID = header_row.index(CommentIDName)


        csvwriter.writerow(header_row)





        for row in csvreader:

            DeltaAwarded = "0"
            DeltaAwarder = ""
            DeltaAwardText = ""


            #if the comment can be mapped back to a submission author

            if CommentsToSubmissionLinks[row[CommentID]] in SubmissionsToAuthorsLinks.keys():

                
                Submission_Author = SubmissionsToAuthorsLinks[CommentsToSubmissionLinks[row[CommentID]]]


                

                #if the comment is a top-level comment...
                if row[CommentParentID].split('_')[0] == 't3':

                    #...and it is a top-level delta, or it's not contained in the low-level deltas
                    if (row[CommentID] in TopLevelDeltaComments) or (row[CommentID] not in LowLevelDeltaComments):

                        #...and the comment hasn't been deleted, and it's not an automod
                        if (row[CommentBodyID] not in comments_to_omit) and (row[CommentAuthorID] not in authors_to_omit):

                            #...and the comment isn't one made by the submission author themselves
                            if row[CommentAuthorID] != Submission_Author:


                                if row[CommentID] in TopLevelDeltaComments:

                                    DeltaAwarded = "1"
                                    DeltaAwarder = WhoAwardedDeltas[row[CommentID]]
                                    DeltaAwardText = DeltaAwardTextDict[row[CommentID]]

                                row_to_write = row
                                row_to_write[CommentParentID] = row_to_write[CommentParentID].replace('t3_', '')
                                row_to_write.append(DeltaAwarded)
                                row_to_write.append(DeltaAwarder)
                                row_to_write.append(DeltaAwardText)
                                row_to_write.append(Submission_Author)

                                for replacement in Replacements:
                                    row_to_write[CommentBodyID] = row_to_write[CommentBodyID].replace(replacement[0], replacement[1])

                                csvwriter.writerow(row_to_write)
                                

                

            



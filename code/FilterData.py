import csv

#Slims down file that's below into only useful stuff
input_file = "/home/shared/CMV/01 - RawData/01 - RawData/2019-02-11 - Extracted Reddit Data - Comments - MetaData.csv"
lines_to_read = -1
with open(input_file, mode='r', encoding="utf-8") as csv_file:
    file = open("Slimmed_Comments_MetaData", mode='w', encoding="utf-8")
    writer = csv.writer(file, dialect='excel', delimiter=',')

    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Initial columns being read: {", ".join(row)}\n')
            writer.writerow(['name','parent_id','author','link_id'])
            line_count += 1
        if row["author"] != "[deleted]":
            writer.writerow([row['name'],row['parent_id'],row['author'],row['link_id']])

        line_count += 1
        if (line_count >= lines_to_read) and (lines_to_read>0):
            break;

    file.close()


Replacements = [["&nbsp;", " "], ["&iexcl;", "¡"], ["&cent;", "¢"], ["&pound;", "£"], ["&curren;", "¤"], ["&yen;", "¥"], ["&brvbar;", "¦"], ["&sect;", "§"], ["&uml;", "¨"], ["&copy;", "©"], ["&ordf;", "ª"], ["&laquo;", "«"], ["&not;", "¬"], ["&reg;", "®"], ["&macr;", "¯"], ["&deg;", "°"], ["&plusmn;", "±"], ["&sup2;", "²"], ["&sup3;", "³"], ["&acute;", "´"], ["&micro;", "µ"], ["&para;", "¶"], ["&cedil;", "¸"], ["&sup1;", "¹"], ["&ordm;", "º"], ["&raquo;", "»"], ["&frac14;", "¼"], ["&frac12;", "½"], ["&frac34;", "¾"], ["&iquest;", "¿"], ["&times;", "×"], ["&divide;", "÷"], ["&Agrave;", "À"], ["&Aacute;", "Á"], ["&Acirc;", "Â"], ["&Atilde;", "Ã"], ["&Auml;", "Ä"], ["&Aring;", "Å"], ["&AElig;", "Æ"], ["&Ccedil;", "Ç"], ["&Egrave;", "È"], ["&Eacute;", "É"], ["&Ecirc;", "Ê"], ["&Euml;", "Ë"], ["&Igrave;", "Ì"], ["&Iacute;", "Í"], ["&Icirc;", "Î"], ["&Iuml;", "Ï"], ["&ETH;", "Ð"], ["&Ntilde;", "Ñ"], ["&Ograve;", "Ò"], ["&Oacute;", "Ó"], ["&Ocirc;", "Ô"], ["&Otilde;", "Õ"], ["&Ouml;", "Ö"], ["&Oslash;", "Ø"], ["&Ugrave;", "Ù"], ["&Uacute;", "Ú"], ["&Ucirc;", "Û"], ["&Uuml;", "Ü"], ["&Yacute;", "Ý"], ["&THORN;", "Þ"], ["&szlig;", "ß"], ["&agrave;", "à"], ["&aacute;", "á"], ["&acirc;", "â"], ["&atilde;", "ã"], ["&auml;", "ä"], ["&aring;", "å"], ["&aelig;", "æ"], ["&ccedil;", "ç"], ["&egrave;", "è"], ["&eacute;", "é"], ["&ecirc;", "ê"], ["&euml;", "ë"], ["&igrave;", "ì"], ["&iacute;", "í"], ["&icirc;", "î"], ["&iuml;", "ï"], ["&eth;", "ð"], ["&ntilde;", "ñ"], ["&ograve;", "ò"], ["&oacute;", "ó"], ["&ocirc;", "ô"], ["&otilde;", "õ"], ["&ouml;", "ö"], ["&oslash;", "ø"], ["&ugrave;", "ù"], ["&uacute;", "ú"], ["&ucirc;", "û"], ["&uuml;", "ü"], ["&yacute;", "ý"], ["&thorn;", "þ"], ["&yuml;", "ÿ"], ["&amp;", "&"], ["&lt;", "<"], ["&gt;", ">"], ["&OElig;", "Œ"], ["&oelig;", "œ"], ["&Scaron;", "Š"], ["&scaron;", "š"], ["&Yuml;", "Ÿ"], ["&fnof;", "ƒ"], ["&circ;", "ˆ"], ["&tilde;", "˜"], ["&ndash;", "–"], ["&mdash;", "—"], ["&lsquo;", "‘"], ["&rsquo;", "’"], ["&sbquo;", "‚"], ["&ldquo;", "“"], ["&rdquo;", "”"], ["&bdquo;", "„"], ["&dagger;", "†"], ["&Dagger;", "‡"], ["&bull;", "•"], ["&hellip;", "…"], ["&permil;", "‰"], ["&lsaquo;", "‹"], ["&rsaquo;", "›"], ["&euro;", "€"], ["&trade;", "™"]]


#Slims down file that's below into only useful stuff
input_file = "/home/shared/CMV/RawData/2019-02-11 - Extracted Reddit Data - Comments - TextData.csv"
lines_to_read = -1
with open(input_file, mode='r', encoding="utf-8") as csv_file:
    file = open("/home/shared/CMV/Slimmed_Comments_TextData", mode='w', encoding="utf-8")
    writer = csv.writer(file, dialect='excel', delimiter=',')

    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Initial columns being read: {", ".join(row)}\n')
            writer.writerow(['author','id','parent_id','body'])
            line_count += 1
        if row["author"] != "[deleted]":
            body = row['body']

            for replacement in Replacements:
                body = body.replace(replacement[0], replacement[1])

            writer.writerow([row['author'],row['id'],row['parent_id'],body])

        line_count += 1
        if (line_count >= lines_to_read) and (lines_to_read>0):
            break;
    file.close()

#Slims down file that's below into only useful stuff
input_file = "/home/shared/CMV/RawData/2019-02-11 - Extracted Reddit Data - Submissions - MetaData.csv"
lines_to_read = -1
with open(input_file, mode='r', encoding="utf-8") as csv_file:
    file = open("/home/shared/CMV/Slimmed_Submissions_MetaData", mode='w', encoding="utf-8")
    writer = csv.writer(file, dialect='excel', delimiter=',')

    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Initial columns being read: {", ".join(row)}\n')
            writer.writerow(['url','id','author'])
            line_count += 1
        if row["author"] != "[deleted]":
            writer.writerow([row['url'],row['id'],row['author']])

        line_count += 1
        if (line_count >= lines_to_read) and (lines_to_read>0):
            break;

    file.close()

#Slims down file that's below into only useful stuff
input_file = "/home/shared/CMV/RawData/2019-02-11 - Extracted Reddit Data - Submissions - TextData.csv"
lines_to_read = -1
with open(input_file, mode='r', encoding="utf-8") as csv_file:
    file = open("/home/shared/CMV/Slimmed_Submissions_TextData", mode='w', encoding="utf-8")
    writer = csv.writer(file, dialect='excel', delimiter=',')

    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Initial columns being read: {", ".join(row)}\n')
            writer.writerow(['author','id','title','selftext'])
            line_count += 1
        if row["author"] != "[deleted]":
            writer.writerow([row['author'],row['id'],row['title'],row['selftext']])

        line_count += 1
        if (line_count >= lines_to_read) and (lines_to_read>0):
            break;

    file.close()

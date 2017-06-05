import glob
import sys

def read_file(file_name):
    file_ = open(file_name, "r")
    end_of_sentence = True
    line = file_.readline()
    content = []
    num_chars = 0
    while line != "":
        line = line.replace('\t', ' ')
        if line == '\n':
            if end_of_sentence:
                content.extend('\n\n')
            #else:
            #    content.append(' ')
                num_chars += 1
            while line == '\n':
                line = file_.readline()
        #else:

        try:
            end_of_sentence = (line[-2] in '.!?')
        except:
            end_of_sentence = False
        line = line[:-1] + ' '
        content.extend([x for x in line if ord(x) < 128])
        num_chars += len(line)
        line = file_.readline()
    file_.close()
    return content

def read_data_files_from_folder(directory, validation = 0.1):

    codetext   = []
    bookranges = []

    file_start = 0
    for file_name in glob.glob(directory, recursive = True):
        file_ = open(file_name, "r")
        print("Loading file: " + file_name)
        chars = read_file(file_name)
        codetext.extend(chars)
        num_chars = len(chars)
        bookranges.append({"start": file_start,
                           "end": file_start + num_chars,
                           "name": file_name.rsplit("/", 1)[-1]})
        file_start += num_chars

    if len(bookranges) == 0:
        sys.exit("No training data has been found. Aborting.")



    # For validation, we use 90k of text, it that number does not exceed 10% of the total dataset
    # or one book (the content of one file).  If validdation parameter is set and is a floating
    # point number, then that fraction is used as the cap.

    total_length          = len(codetext)
    validation_length_max = 90 * 1024
    validation_length_max = min(validation_length_max, validation * total_length)


    #validation_fraction

    # For validation, use roughly 90K of text,
    # but no more than 10% of the entire text
    # and no more than 1 book in 5 => no validation at all for 5 files or fewer.

    # 10% of the text is how many files ?
    #total_length = len(codetext)
    #validation_length = 0
    #nb_books1 = 0
    #for book in reversed(bookranges):
    #    validation_len += book["end"]-book["start"]
    #    nb_books1 += 1
    #    if validation_len > total_len // 10:
    #        break
#
#    # 90K of text is how many books ?
#    validation_len = 0
#    nb_books2 = 0
#    for book in reversed(bookranges):
#        validation_len += book["end"]-book["start"]
#        nb_books2 += 1
#        if validation_len > 90*1024:
#            break
#
#    # 20% of the books is how many books ?
#    nb_books3 = len(bookranges) // 5
#
#    # pick the smallest
#    nb_books = min(nb_books1, nb_books2, nb_books3)
#
    #if nb_books == 0 or not validation:
    #    cutoff = len(codetext)
    #else:
    #    cutoff = bookranges[-nb_books]["start"]
    #valitext = codetext[cutoff:]
    #codetext = codetext[:cutoff]
    return codetext, [], bookranges


if __name__ == '__main__':
    text, valitext, ranges = read_data_files_from_folder('text_generator/training_data/harry_potter/1 - *.txt')
    print(''.join(text[:10000]))
    #print(len(text), len(valitext), len(set(text)))

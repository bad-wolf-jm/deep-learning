import glob
import sys

def read_file(file_name):
    file_ = open(file_name, "r")
    end_of_sentence = True
    line = file_.readline()
    content = []
    num_chars = 0
    while line != "":
        if line == '\n':
            if end_of_sentence:
                content.append('\n')
                num_chars += 1
            while line == '\n':
                line = file_.readline()

        else:
            line = line[:-1]
            try:
                end_of_sentence = (line[-1] == '.')
            except:
                end_of_sentence = False
            content.extend([x for x in line])
            num_chars += len(line)
            line = file_.readline()
    file_.close()
    return content

def read_data_files_from_folder(directory, validation = True):

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

    # For validation, use roughly 90K of text,
    # but no more than 10% of the entire text
    # and no more than 1 book in 5 => no validation at all for 5 files or fewer.

    # 10% of the text is how many files ?
    total_len = len(codetext)
    validation_len = 0
    nb_books1 = 0
    for book in reversed(bookranges):
        validation_len += book["end"]-book["start"]
        nb_books1 += 1
        if validation_len > total_len // 10:
            break

    # 90K of text is how many books ?
    validation_len = 0
    nb_books2 = 0
    for book in reversed(bookranges):
        validation_len += book["end"]-book["start"]
        nb_books2 += 1
        if validation_len > 90*1024:
            break

    # 20% of the books is how many books ?
    nb_books3 = len(bookranges) // 5

    # pick the smallest
    nb_books = min(nb_books1, nb_books2, nb_books3)

    if nb_books == 0 or not validation:
        cutoff = len(codetext)
    else:
        cutoff = bookranges[-nb_books]["start"]
    valitext = codetext[cutoff:]
    codetext = codetext[:cutoff]
    return codetext, valitext, bookranges


if __name__ == '__main__':
    text, valitext, ranges = read_data_files_from_folder('training_data/gcc-master/**/*.c')
    print(len(text), len(valitext), len(set(text)))

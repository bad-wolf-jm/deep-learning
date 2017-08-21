import os


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), 'gru_sort_outputs')

    files = {}
    for file_name in os.listdir(path):
        _, _, i = file_name.split('-')
        i = int(i)
        files[i] = os.path.join(path, file_name)
    files_in_order = sorted(files.keys())
    accuracies = []
    for index in files_in_order:
        file_name = files[index]
        print(file_name, end="")
        with open(file_name) as data_file:
            lines = data_file.readlines()
            accs = 0
            for line in lines:
                input_, output_ = line.split(';')
                input_ = [int(x) for x in input_.split(',')]
                output_ = [int(x) for x in output_.split(',')]
                N = len(input_)
                I = 0
                for x, y in zip(sorted(input_), output_):
                    if x == y:
                        I += 1
                accs += float(I) / N
            accuracies.append(accs / len(lines))
            print("  ", accs / len(lines))

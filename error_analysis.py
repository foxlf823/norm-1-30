import codecs
import json
from collections import Counter, OrderedDict

# compute the occurrence numbers of correct and incorrect entities
def error1():
    with codecs.open("sorted_correct_entities.txt", 'r', 'UTF-8') as fp:
        str1 = fp.read()

    sorted_correct_entities = json.loads(str1)

    correct_entities_occur_number = OrderedDict()
    for k,v in sorted_correct_entities.items():
        if v in correct_entities_occur_number:
            correct_entities_occur_number[v] += 1
        else:
            correct_entities_occur_number[v] = 1

    with codecs.open("sorted_wrong_entities.txt", 'r', 'UTF-8') as fp:
        str1 = fp.read()

    sorted_wrong_entities = json.loads(str1)

    wrong_entities_occur_number = OrderedDict()
    for k,v in sorted_wrong_entities.items():
        if v in wrong_entities_occur_number:
            wrong_entities_occur_number[v] += 1
        else:
            wrong_entities_occur_number[v] = 1

    with codecs.open("sorted_correct_entities.csv", 'w', 'UTF-8') as fp:

        for k,v in sorted_correct_entities.items():
            fp.write(k+","+str(v)+"\n")

        fp.write("\n")

        for k,v in correct_entities_occur_number.items():
            fp.write(str(k) + "," + str(v) + "\n")

    with codecs.open("sorted_wrong_entities.csv", 'w', 'UTF-8') as fp:

        for k,v in sorted_wrong_entities.items():
            fp.write(k+","+str(v)+"\n")

        fp.write("\n")

        for k, v in wrong_entities_occur_number.items():
            fp.write(str(k) + "," + str(v) + "\n")

# compute the frequency of each word in the entities
def error2():
    with codecs.open("sorted_correct_entities.txt", 'r', 'UTF-8') as fp:
        str1 = fp.read()

    sorted_correct_entities = json.loads(str1)

    correct_word_counter = Counter()
    for k,v in sorted_correct_entities.items():
        words = k.split()
        for word in words:
            if word in correct_word_counter:
                correct_word_counter[word] += v
            else:
                correct_word_counter[word] = v
                
    correct_word_frequency_counter = Counter()
    correct_entities_number = len(sorted_correct_entities)
    for k,v in correct_word_counter.items():
        correct_word_frequency_counter[k] = round(v/correct_entities_number, 2)

    with codecs.open("word_frequency_correct.csv", 'w', 'UTF-8') as fp:

        for k, v in correct_word_frequency_counter.most_common():
            fp.write(k + "," + str(v) + "\n")

    with codecs.open("sorted_wrong_entities.txt", 'r', 'UTF-8') as fp:
        str1 = fp.read()

    sorted_wrong_entities = json.loads(str1)

    wrong_word_counter = Counter()
    for k,v in sorted_wrong_entities.items():
        words = k.split()
        for word in words:
            if word in wrong_word_counter:
                wrong_word_counter[word] += v
            else:
                wrong_word_counter[word] = v
                
    wrong_word_frequency_counter = Counter()
    wrong_entities_number = len(sorted_wrong_entities)
    for k,v in wrong_word_counter.items():
        wrong_word_frequency_counter[k] = round(v/wrong_entities_number, 2)
                
    with codecs.open("word_frequency_wrong.csv", 'w', 'UTF-8') as fp:

        for k, v in wrong_word_frequency_counter.most_common():
            fp.write(k + "," + str(v) + "\n")

    




if __name__ == '__main__':

    error2()

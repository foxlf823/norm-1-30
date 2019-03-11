import codecs
from options import opt


def getStopword_fromFile(file_path):
    s = set()
    with codecs.open(file_path, 'r', 'UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line == u'':
                continue
            s.add(line)

    return s

stop_word = getStopword_fromFile(opt.stopword)
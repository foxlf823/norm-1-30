import codecs
import os
import nltk
import re
import spacy
nlp_tool = spacy.load('en')

import xml.sax
import zipfile

# all text follow word2vec and fasttext format
# lower cased
# number normalized to 0
# utf-8

pattern = re.compile(r'[-_/]+')

def my_split(s):
    text = []
    iter = re.finditer(pattern, s)
    start = 0
    for i in iter:
        if start != i.start():
            text.append(s[start: i.start()])
        text.append(s[i.start(): i.end()])
        start = i.end()
    if start != len(s):
        text.append(s[start: ])
    return text


def my_tokenize(txt):
    # tokens1 = nltk.word_tokenize(txt.replace('"', " "))  # replace due to nltk transfer " to other character, see https://github.com/nltk/nltk/issues/1630
    # tokens2 = []
    # for token1 in tokens1:
    #     token2 = my_split(token1)
    #     tokens2.extend(token2)
    # return tokens2


    document = nlp_tool(txt)
    sentences = []
    for span in document.sents:
        sentence = [document[i] for i in range(span.start, span.end)]

        tokens = []
        for t in sentence:
            token1 = t.text.strip()
            if token1 == '':
                continue
            token2 = my_split(token1)
            tokens.extend(token2)

        sentences.append(tokens)

    return sentences


def normalize_word_digit(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def nlp_process(data):
    sentences = my_tokenize(data)

    sentences_normed = []
    for sent in sentences:

        tokens_normed = []
        for token in sent:
            token = token.lower()
            token = normalize_word_digit(token)
            tokens_normed.append(token)
        sentences_normed.append(tokens_normed)

    return sentences_normed


def ehr_to_text(dir_path, output_file_path, append):

    if append:
        output_fp = codecs.open(output_file_path, 'a', 'UTF-8')
    else:
        output_fp = codecs.open(output_file_path, 'w', 'UTF-8')

    list_dir = os.listdir(dir_path)

    print("total: {}".format(len(list_dir)))

    for file_name in list_dir:
        if file_name.find("DS_Store") != -1:
            continue

        print("processing {}".format(file_name))

        with codecs.open(os.path.join(dir_path, file_name), 'r', 'UTF-8') as fp:
            data = fp.read()

            sentences = nlp_process(data)

            for sent in sentences:
                for token in sent:
                    output_fp.write(token+' ')
                output_fp.write('\n')

    output_fp.close()


def faers_to_text(dir_path, output_file_path, append):

    if append:
        output_fp = codecs.open(output_file_path, 'a', 'UTF-8')
    else:
        output_fp = codecs.open(output_file_path, 'w', 'UTF-8')

    list_dir = os.listdir(dir_path)

    print("total: {}".format(len(list_dir)))

    for file_name in list_dir:
        if file_name.find("DS_Store") != -1:
            continue

        print("processing {}".format(file_name))

        with codecs.open(os.path.join(dir_path, file_name), 'r', 'LATIN_1') as fp:
            count = 1
            data = ''
            for line in fp:
                if count != 1:
                    data += line
                count += 1

            sentences = nlp_process(data)

            for sent in sentences:
                for token in sent:
                    output_fp.write(token + ' ')
                output_fp.write('\n')


    output_fp.close()


def cdr_to_text(dir_path, output_file_path, append):

    if append:
        output_fp = codecs.open(output_file_path, 'a', 'UTF-8')
    else:
        output_fp = codecs.open(output_file_path, 'w', 'UTF-8')

    list_dir = os.listdir(dir_path)

    for file_name in list_dir:
        if file_name.find("DS_Store") != -1:
            continue

        if file_name.find('.PubTator.txt') == -1:
            continue


        with codecs.open(os.path.join(dir_path, file_name), 'r', 'UTF-8') as fp:

            for line in fp:
                _t_position = line.find('|t|')
                if _t_position != -1:
                    id = line[0 : _t_position]
                    print("processing {}".format(id))
                    title = line[_t_position+len('|t|'):]
                    sentences = nlp_process(title)

                    for sent in sentences:
                        for token in sent:
                            output_fp.write(token + ' ')
                        output_fp.write('\n')

                _a_position = line.find('|a|')
                if _a_position != -1:
                    abstract = line[_a_position+len('|a|'):]
                    sentences = nlp_process(abstract)

                    for sent in sentences:
                        for token in sent:
                            output_fp.write(token + ' ')
                        output_fp.write('\n')

    output_fp.close()


def ncbi_disease_to_text(dir_path, output_file_path, append):

    if append:
        output_fp = codecs.open(output_file_path, 'a', 'UTF-8')
    else:
        output_fp = codecs.open(output_file_path, 'w', 'UTF-8')

    list_dir = os.listdir(dir_path)

    for file_name in list_dir:
        if file_name.find("DS_Store") != -1:
            continue

        if file_name.find('.txt') == -1:
            continue


        with codecs.open(os.path.join(dir_path, file_name), 'r', 'UTF-8') as fp:

            for line in fp:
                _t_position = line.find('|t|')
                if _t_position != -1:
                    id = line[0 : _t_position]
                    print("processing {}".format(id))
                    title = line[_t_position+len('|t|'):]
                    sentences = nlp_process(title)

                    for sent in sentences:
                        for token in sent:
                            output_fp.write(token + ' ')
                        output_fp.write('\n')

                _a_position = line.find('|a|')
                if _a_position != -1:
                    abstract = line[_a_position+len('|a|'):]
                    sentences = nlp_process(abstract)

                    for sent in sentences:
                        for token in sent:
                            output_fp.write(token + ' ')
                        output_fp.write('\n')

    output_fp.close()

def ade_to_text(dir_path, output_file_path, append):

    if append:
        output_fp = codecs.open(output_file_path, 'a', 'UTF-8')
    else:
        output_fp = codecs.open(output_file_path, 'w', 'UTF-8')


    with codecs.open(os.path.join(dir_path, "ADE-NEG.txt"), 'r', 'UTF-8') as fp:

        for line in fp:
            _t_position = line.find(' NEG ')
            if _t_position != -1:
                id = line[0 : _t_position]
                print("processing {}".format(id))
                title = line[_t_position+len(' NEG '):]
                sentences = nlp_process(title)

                for sent in sentences:
                    for token in sent:
                        output_fp.write(token + ' ')
                    output_fp.write('\n')

    with codecs.open(os.path.join(dir_path, "DRUG-AE.rel"), 'r', 'UTF-8') as fp:

        for line in fp:
            spliteed = line.split('|')
            print("processing {}".format(spliteed[0]))
            sentences = nlp_process(spliteed[1])

            for sent in sentences:
                for token in sent:
                    output_fp.write(token + ' ')
                output_fp.write('\n')

    output_fp.close()

def clef_to_text(dir_path, output_file_path, append):

    if append:
        output_fp = codecs.open(output_file_path, 'a', 'UTF-8')
    else:
        output_fp = codecs.open(output_file_path, 'w', 'UTF-8')

    list_dir = os.listdir(dir_path)

    print("total: {}".format(len(list_dir)))

    for file_name in list_dir:
        if file_name.find("DS_Store") != -1:
            continue

        if file_name.find(".pipe.txt") != -1:
            continue

        print("processing {}".format(file_name))

        with codecs.open(os.path.join(dir_path, file_name), 'r', 'UTF-8') as fp:
            data = fp.read()

            sentences = nlp_process(data)

            for sent in sentences:
                for token in sent:
                    output_fp.write(token+' ')
                output_fp.write('\n')

    output_fp.close()


def pubmed_to_text(dir_path, output_file_path, append):

    if append:
        output_fp = codecs.open(output_file_path, 'a', 'UTF-8')
    else:
        output_fp = codecs.open(output_file_path, 'w', 'UTF-8')

    list_dir = os.listdir(dir_path)

    for file_name in list_dir:
        if file_name.find("DS_Store") != -1:
            continue


        with codecs.open(os.path.join(dir_path, file_name), 'r', 'UTF-8') as fp:

            for line in fp:
                _t_position = line.find('|t|')
                if _t_position != -1:
                    id = line[0 : _t_position]
                    print("processing {}".format(id))
                    title = line[_t_position+len('|t|'):]
                    sentences = nlp_process(title)

                    for sent in sentences:
                        for token in sent:
                            output_fp.write(token + ' ')
                        output_fp.write('\n')

                _a_position = line.find('|a|')
                if _a_position != -1:
                    abstract = line[_a_position+len('|a|'):]
                    sentences = nlp_process(abstract)

                    for sent in sentences:
                        for token in sent:
                            output_fp.write(token + ' ')
                        output_fp.write('\n')

    output_fp.close()


class DrugBankHandler( xml.sax.ContentHandler ):
    def __init__(self, output_file_path, append):
        self.currentTag = ""
        self.parentTag = []
        self.currentData = ''
        self.append = append
        self.output_file_path = output_file_path

    def startDocument(self):

        if self.append:
            self.output_fp = codecs.open(self.output_file_path, 'a', 'UTF-8')
        else:
            self.output_fp = codecs.open(self.output_file_path, 'w', 'UTF-8')

    def endDocument(self):
        self.output_fp.close()

    def startElement(self, tag, attributes):
        if self.currentTag != '':
            self.parentTag.append(self.currentTag)

        self.currentTag = tag


    def endElement(self, tag):
        if len(self.parentTag) != 0:
            self.currentTag = self.parentTag[-1]
            self.parentTag.pop()
        else:
            self.currentTag = ''


    def characters(self, content):

        if len(self.parentTag) == 2 and self.parentTag[-1] == 'drug' and self.currentTag == 'name':
            print("processing {}".format(content))
        elif len(self.parentTag)>0 and self.parentTag[-1] == 'drug' and self.currentTag == 'description':
            if content.strip() == '':
                return
            sentences = nlp_process(content)

            for sent in sentences:
                for token in sent:
                    self.output_fp.write(token + ' ')
                self.output_fp.write('\n')
        elif len(self.parentTag)>0 and self.parentTag[-1] == 'drug' and self.currentTag == 'indication':
            if content.strip() == '':
                return
            sentences = nlp_process(content)

            for sent in sentences:
                for token in sent:
                    self.output_fp.write(token + ' ')
                self.output_fp.write('\n')
        elif len(self.parentTag)>0 and self.parentTag[-1] == 'drug' and self.currentTag == 'pharmacodynamics':
            if content.strip() == '':
                return
            sentences = nlp_process(content)

            for sent in sentences:
                for token in sent:
                    self.output_fp.write(token + ' ')
                self.output_fp.write('\n')
        elif len(self.parentTag)>0 and self.parentTag[-1] == 'drug' and self.currentTag == 'mechanism-of-action':
            if content.strip() == '':
                return
            sentences = nlp_process(content)

            for sent in sentences:
                for token in sent:
                    self.output_fp.write(token + ' ')
                self.output_fp.write('\n')
        elif len(self.parentTag)>0 and self.parentTag[-1] == 'drug' and self.currentTag == 'toxicity':
            if content.strip() == '':
                return
            sentences = nlp_process(content)

            for sent in sentences:
                for token in sent:
                    self.output_fp.write(token + ' ')
                self.output_fp.write('\n')
        elif len(self.parentTag)>1 and self.parentTag[-2] == 'drug-interactions' and self.parentTag[-1] == 'drug-interaction' and self.currentTag == 'description':
            if content.strip() == '':
                return
            sentences = nlp_process(content)

            for sent in sentences:
                for token in sent:
                    self.output_fp.write(token + ' ')
                self.output_fp.write('\n')


class DailyMedHandler( xml.sax.ContentHandler ):
    def __init__(self, output_fp):
        self.output_fp = output_fp
        self.currentTag = ""
        self.parentTag = []

    def startDocument(self):
        pass

    def endDocument(self):
        pass

    def startElement(self, tag, attributes):
        if self.currentTag != '':
            self.parentTag.append(self.currentTag)

        self.currentTag = tag


    def endElement(self, tag):
        if len(self.parentTag) != 0:
            self.currentTag = self.parentTag[-1]
            self.parentTag.pop()
        else:
            self.currentTag = ''


    def characters(self, content):

        if len(self.parentTag)>0 and self.parentTag[-1] == 'text' and self.currentTag == 'paragraph':
            if content.strip() == '':
                return

            if re.search(r'[a-zA-Z]+', content) == None:
                return

            sentences = nlp_process(content)

            for sent in sentences:
                has_alpha = False
                for token in sent:
                    if re.search(r'[a-zA-Z]+', token) != None:
                        has_alpha = True
                        break

                if has_alpha == False:
                    continue

                for token in sent:
                    self.output_fp.write(token + ' ')
                self.output_fp.write('\n')

        elif len(self.parentTag)>1 and self.parentTag[-2] == 'text' and self.parentTag[-1] == 'list' and self.currentTag == 'item':
            if content.strip() == '':
                return

            if re.search(r'[a-zA-Z]+', content) == None:
                return

            sentences = nlp_process(content)

            for sent in sentences:
                has_alpha = False
                for token in sent:
                    if re.search(r'[a-zA-Z]+', token) != None:
                        has_alpha = True
                        break

                if has_alpha == False:
                    continue

                for token in sent:
                    self.output_fp.write(token + ' ')
                self.output_fp.write('\n')


def dailymed_parse_one_xml(file_path, output_file_path, parser):
    output_fp = codecs.open(output_file_path, 'w', 'UTF-8')

    Handler = DailyMedHandler(output_fp)
    parser.setContentHandler(Handler)

    parser.parse(os.path.join(file_path))

    output_fp.close()

def dailymed_to_text(dir_path, output_file_path, append):
    if append:
        output_fp = codecs.open(output_file_path, 'a', 'UTF-8')
    else:
        output_fp = codecs.open(output_file_path, 'w', 'UTF-8')

    Handler = DailyMedHandler(output_fp)

    list_dir = os.listdir(dir_path)

    print("total: {}".format(len(list_dir)))

    for file_name in list_dir:

        if file_name.find('.zip') == -1:
            continue

        print("processing {}".format(file_name))

        z = zipfile.ZipFile(os.path.join(dir_path, file_name), "r")
        for zip_filename in z.namelist():
            if zip_filename.find('.xml') != -1:
                xml_content = z.read(zip_filename).decode('utf-8')
                xml.sax.parseString(xml_content, Handler)

    output_fp.close()


class FdaXmlHandler( xml.sax.ContentHandler ):

    def __init__(self, output_fp):
        self.currentTag = ""
        self.parentTag = []
        self.output_fp = output_fp

    def startDocument(self):
        pass

    def endDocument(self):
        self.currentTag = ""
        self.parentTag = []

    def startElement(self, tag, attributes):
        if self.currentTag != '':
            self.parentTag.append(self.currentTag)

        self.currentTag = tag


    def endElement(self, tag):
        if len(self.parentTag) != 0:
            self.currentTag = self.parentTag[-1]
            self.parentTag.pop()
        else:
            self.currentTag = ''


    def characters(self, content):

        if self.currentTag == 'Section':
            if content.strip() == '':
                return

            if re.search(r'[a-zA-Z]+', content) == None:
                return

            sentences = nlp_process(content)

            for sent in sentences:
                has_alpha = False
                for token in sent:
                    if re.search(r'[a-zA-Z]+', token) != None:
                        has_alpha = True
                        break

                if has_alpha == False:
                    continue

                for token in sent:
                    self.output_fp.write(token + ' ')
                self.output_fp.write('\n')

def tac2017_to_text(dir_path, output_file_path, append):
    if append:
        output_fp = codecs.open(output_file_path, 'a', 'UTF-8')
    else:
        output_fp = codecs.open(output_file_path, 'w', 'UTF-8')

    Handler = FdaXmlHandler(output_fp)

    list_dir = os.listdir(dir_path)

    print("total: {}".format(len(list_dir)))

    for file_name in list_dir:

        if file_name.find('.xml') == -1:
            continue

        print("processing {}".format(file_name))

        xml.sax.parse(os.path.join(dir_path, file_name), Handler)

    output_fp.close()


if __name__ == '__main__':



    # ehr: made, cardio, hypoglecimia
    # output_file = '/Users/feili/resource/data_to_train_emb/ehr.txt'
    # ehr_to_text('/Users/feili/Desktop/umass/MADE/MADE-1.0/corpus', output_file, False)
    # ehr_to_text('/Users/feili/Desktop/umass/MADE/made_test_data/corpus', output_file, True)
    # ehr_to_text('/Users/feili/Desktop/umass/bioC_data/Cardio_train/corpus', output_file, True)
    # ehr_to_text('/Users/feili/Desktop/umass/bioC_data/Cardio_test/corpus', output_file, True)
    # ehr_to_text('/Users/feili/Desktop/umass/hypoglycemia/ehost_annotations_hypoglycemia_201807/ehost_annotations_hypoglycemia_201807/corpus',
    #             output_file, True)

    # FAERS
    # output_file = '/Users/feili/resource/data_to_train_emb/faers.txt'
    # faers_to_text('/Users/feili/Desktop/umass/FAERS_122_Reports/Nadya_51/aers51', output_file, False)
    # faers_to_text('/Users/feili/Desktop/umass/FAERS_122_Reports/Nadya-11/aers11', output_file, True)
    # faers_to_text('/Users/feili/Desktop/umass/FAERS_122_Reports/Nadya-48/aers48', output_file, True)

    # cdr
    # output_file = '/Users/feili/resource/data_to_train_emb/cdr.txt'
    # cdr_to_text('/Users/feili/old_file/v/cdr/CDR_Data/CDR.Corpus.v010516' ,output_file, False)

    # ncbi disease
    # output_file = '/Users/feili/resource/data_to_train_emb/ncbi_disease.txt'
    # ncbi_disease_to_text('/Users/feili/old_file/NCBI disease corpus', output_file, False)

    # ade
    # output_file = '/Users/feili/resource/data_to_train_emb/ade.txt'
    # ade_to_text('/Users/feili/old_file/ADE-Corpus-V2', output_file, False)

    # clef 2013
    # output_file = '/Users/feili/resource/data_to_train_emb/clef.txt'
    # clef_to_text('/Users/feili/old_file/clef/2013/clef2013/task1train/ALLREPORTS', output_file, False)
    # clef_to_text('/Users/feili/old_file/clef/2013/clef2013/task1test/ALLREPORTS', output_file, True)

    # pubmed
    # output_file = '/Users/feili/resource/data_to_train_emb/pubmed.txt'
    # pubmed_to_text('/Users/feili/resource/data_to_train_emb/pubmed_ade', output_file, False)

    # drugbank
    # output_file = '/Users/feili/resource/data_to_train_emb/drugbank.txt'
    # parser = xml.sax.make_parser()
    # parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    # Handler = DrugBankHandler(output_file, False)
    # parser.setContentHandler(Handler)
    # parser.parse("/Users/feili/resource/drugbank_database.xml")

    # dailymed
    # output_file = '/Users/feili/resource/data_to_train_emb/dailymed.txt'
    # dailymed_to_text("/Users/feili/resource/dm_spl_monthly_update_oct2018/prescription", output_file, False)
    # dailymed_to_text("/Users/feili/resource/dm_spl_monthly_update_oct2018/otc", output_file, True)
    # dailymed_parse_one_xml('/Users/feili/Downloads/AFINITOR11232018/beceb18a-7957-4a80-bcc1-b4a0b05ef106.xml', output_file, parser)

    # tac 2017
    # output_file = '/Users/feili/resource/data_to_train_emb/tac2017.txt'
    # tac2017_to_text("/Users/feili/dataset/tac_2017_ade/train_xml", output_file, False)
    # tac2017_to_text("/Users/feili/dataset/tac_2017_ade/unannotated_xml", output_file, True)

    # training set
    # output_file = '/Users/feili/resource/data_to_train_emb/fda2018.txt'
    # tac2017_to_text("/Users/feili/dataset/ADE Eval Shared Resources/ose_xml_training_20181101", output_file, False)

    # pubmed meddra
    # output_file = '/Users/feili/resource/data_to_train_emb/pubmed_meddra.txt'
    # pubmed_to_text('/Users/feili/resource/pubmed_meddra', output_file, False)

    # pubmed snomed
    # output_file = '/Users/feili/resource/data_to_train_emb/pubmed_snomed.txt'
    # pubmed_to_text('/Users/feili/resource/pubmed_snomed', output_file, False)

    # test set
    output_file = '/Users/feili/resource/data_to_train_emb/fda2018_test.txt'
    tac2017_to_text("/Users/feili/dataset/ADE Eval Shared Resources/UnannotatedTestCorpus", output_file, False)



    pass
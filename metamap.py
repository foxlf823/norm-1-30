#!/usr/bin/python
# -*- coding: UTF-8 -*-

from os import listdir
from os.path import isfile, join
import os
import codecs
from my_utils import makedir_and_clear
from data_structure import Entity,Document
import re


def apply_metamap_to(input_dir, output_dir):
    makedir_and_clear(output_dir)

    for input_file_name in listdir(input_dir):
        input_file_path = join(input_dir, input_file_name)
        if input_file_name.rfind('.') == -1:
            output_file_name = input_file_name + ".field.txt"
        else:
            output_file_name = input_file_name[0:input_file_name.rfind('.')]+".field.txt"
        output_file_path = join(output_dir, output_file_name)
        os.system(
            '/Users/feili/tools/metamap/public_mm/bin/metamap -y -I -N --blanklines 0 -R SNOMEDCT_US,MDR -J acab,anab,comd,cgab,dsyn,emod,fndg,inpo,mobd,neop,patf,sosy {} {}'.format(
                input_file_path, output_file_path))
        # os.system('/Users/feili/tools/metamap/public_mm/bin/metamap -y -I -N --blanklines 0 -R SNOMEDCT_US -J acab,anab,comd,cgab,dsyn,emod,fndg,inpo,mobd,neop,patf,sosy {} {}'.format(input_file_path, output_file_path))
        # os.system('/Users/feili/tools/metamap/public_mm/bin/metamap -y -I -N --blanklines 0 -J acab,anab,comd,cgab,dsyn,emod,fndg,inpo,mobd,neop,patf,sosy {} {}'.format(input_file_path, output_file_path))

def load_metamap_result_from_file(file_path):
    re_brackets = re.compile(r'\[[0-9|/]+\]')
    document = Document()
    entities = []
    with codecs.open(file_path, 'r', 'UTF-8') as fp:
        for line in fp.readlines():
            fields = line.strip().split(u"|")

            if fields[1] != u'MMI':
                continue

            ID = fields[0] # Unique identifier used to identify text being processed. If no identifier is found in the text, 00000000 will be displayed
            MMI = fields[1] # Always MMI
            Score = fields[2] # MetaMap Indexing (MMI) score with a maximum score of 1000.00
            UMLS_Prefer_Name = fields[3] # The UMLS preferred name for the UMLS concept
            UMLS_ID = fields[4] # The CUI for the identified UMLS concept.
            Semantic_Type_List = fields[5] # Comma-separated list of Semantic Type abbreviations
            Trigger_Information = fields[6] # Comma separated sextuple showing what triggered MMI to identify this UMLS concept
            Location = fields[7] # Summarizes where UMLS concept was found. TI – Title, AB – Abstract, TX – Free Text, TI;AB – Title and Abstract
            Positional_Information = fields[8] # Semicolon-separated list of positional-information terns, showing StartPos, slash (/), and Length of each trigger identified in the Trigger Information field
            Treecode = fields[9] # Semicolon-separated list of any MeSH treecode


            triggers = Trigger_Information[1:-1].split(u",\"")
            spans = Positional_Information.split(u";")
            if len(triggers) != len(spans):
                raise RuntimeError("the number of triggers is not equal to that of spans: {} in {}".format(UMLS_ID, file_path[file_path.rfind('/')+1:]))

            for idx, span in enumerate(spans):
                bracket_spans = re_brackets.findall(span)
                if len(bracket_spans) == 0: # simple form
                    if span.find(u',') != -1:
                        print("ignore non-continuous form of Positional_Information: {} in {}".format(triggers[idx],
                                                                                                  file_path[
                                                                                                  file_path.rfind(
                                                                                                      '/') + 1:]))
                        continue


                    tmps = span.split(u"/")
                    entity = Entity()
                    entity.spans.append([int(tmps[0]), int(tmps[0]) + int(tmps[1])])
                    entity.norm_ids.append(str(UMLS_ID))
                    # "B cell lymphoma"-tx-5-"B cell lymphoma"-noun-0
                    tmps = triggers[idx].split(u"-")

                    if tmps[3].find('"') == -1:
                        print("ignore non-string entity: {} in {}".format(tmps[3],
                                                                                                  file_path[
                                                                                                  file_path.rfind(
                                                                                                      '/') + 1:]))
                        continue


                    entity.name = tmps[3][1:-1] # remove ""

                    entities.append(entity)
                else:
                    for bracket_span in bracket_spans:
                        if bracket_span.find(u',') != -1:
                            print("ignore non-continuous form of Positional_Information: {} in {}".format(triggers[idx],
                                                                                                      file_path[
                                                                                                      file_path.rfind(
                                                                                                          '/') + 1:]))
                            continue

                        tmps = bracket_span[1:-1].split(u"/")
                        entity = Entity()
                        entity.spans.append([int(tmps[0]), int(tmps[0]) + int(tmps[1])])
                        entity.norm_ids.append(str(UMLS_ID))
                        # "B cell lymphoma"-tx-5-"B cell lymphoma"-noun-0
                        tmps = triggers[idx].split(u"-")

                        if tmps[3].find('"') == -1:
                            print("ignore non-string entity: {} in {}".format(tmps[3],
                                                                              file_path[
                                                                              file_path.rfind(
                                                                                  '/') + 1:]))
                            continue

                        entity.name = tmps[3][1:-1]

                        entities.append(entity)


    document.entities = entities
    return document


if __name__=="__main__":

# ./bin/metamap -y -I -N --blanklines 0  29_2011-11-15+OC.txt 29_2011-11-15+OC.field.txt
#     apply_metamap_to("/Users/feili/Desktop/umass/CancerADE_SnoM_30Oct2017_test/txt", "/Users/feili/Desktop/umass/CancerADE_SnoM_30Oct2017_test/metamap")

# load_metamap_result_from_file("/Users/feili/Desktop/umass/CancerADE_SnoM_30Oct2017_test/metamap/29_2011-11-15+OC.field.txt")
#     load_metamap_result_from_file(
#     "/Users/feili/Desktop/umass/bioC_data/other/cardio_data/metamap/1001_266.field.txt")


    pass
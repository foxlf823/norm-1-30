
import codecs
from my_utils import setList
from data import loadData
from options import opt

class UMLS_Concept:
    def __init__(self):
        self.cui = None
        self.codes = []
        self.names = []



def load_umls_MRCONSO(file_path):
    UMLS_dict = {} # cui -> codes
    UMLS_dict_reverse = {} # codes -> cui

    with codecs.open(file_path, 'r', 'UTF-8') as fp:
        for line in fp.readlines():
            fields = line.strip().split(u"|")
            CUI = fields[0] # Unique identifier for concept
            LAT = fields[1] # Language of term
            TS = fields[2] # Term status, P - Preferred for CUI, S - Non-Preferred
            LUI = fields[3] # Unique identifier for term
            STT = fields[4] # String type, PF-Preferred form of term
            SUI = fields[5] # Unique identifier for string
            ISPREF = fields[6] # Atom status - preferred (Y) or not (N) for this string within this concept
            AUI = fields[7] # Unique identifier for atom
            SAUI = fields[8] # Source asserted atom identifier
            SCUI = fields[9] # Source asserted concept identifier
            SDUI = fields[10] # Source asserted descriptor identifier
            SAB = fields[11] # Abbreviated source name
            TTY = fields[12] # Abbreviation for term type in source vocabulary, for example PN (Metathesaurus Preferred Name) or CD (Clinical Drug).
            CODE = fields[13] # Most useful source asserted identifier
            STR = fields[14] # String
            SRL = fields[15] # Source restriction level, 0-No additional restrictions; general terms of the license agreement apply, 9-General terms + SNOMED CT Affiliate License in Appendix 2
            SUPPRESS = fields[16] # Suppressible flag. O: All obsolete content, N: None of the above
            CVF = fields[17] # Content View Flag, Bit field used to flag rows included in Content View.

            if CUI in UMLS_dict:
                c = UMLS_dict[CUI]
                setList(c.names, STR)
                setList(c.codes, str(CODE))
            else:
                c = UMLS_Concept()
                c.cui = str(CUI)
                setList(c.names, STR)
                setList(c.codes, str(CODE))
                UMLS_dict[c.cui] = c

            if CODE in UMLS_dict_reverse:
                cui_list = UMLS_dict_reverse[CODE]
                setList(cui_list, CUI)
            else:
                cui_list = [CUI]
                UMLS_dict_reverse[CODE] = cui_list


    return UMLS_dict, UMLS_dict_reverse


def make_debug_dict():

    train_data = loadData('./sample', True, opt.types, opt.type_filter)

    out = codecs.open('./umls_dict_debug.txt', 'w', 'UTF-8')

    with codecs.open('/Users/feili/UMLS/2016AA_Snomed_Meddra/META/MRCONSO.RRF', 'r', 'UTF-8') as fp:
        for line in fp.readlines():
            fields = line.strip().split(u"|")
            CUI = fields[0] # Unique identifier for concept
            LAT = fields[1] # Language of term
            TS = fields[2] # Term status, P - Preferred for CUI, S - Non-Preferred
            LUI = fields[3] # Unique identifier for term
            STT = fields[4] # String type, PF-Preferred form of term
            SUI = fields[5] # Unique identifier for string
            ISPREF = fields[6] # Atom status - preferred (Y) or not (N) for this string within this concept
            AUI = fields[7] # Unique identifier for atom
            SAUI = fields[8] # Source asserted atom identifier
            SCUI = fields[9] # Source asserted concept identifier
            SDUI = fields[10] # Source asserted descriptor identifier
            SAB = fields[11] # Abbreviated source name
            TTY = fields[12] # Abbreviation for term type in source vocabulary, for example PN (Metathesaurus Preferred Name) or CD (Clinical Drug).
            CODE = fields[13] # Most useful source asserted identifier
            STR = fields[14] # String
            SRL = fields[15] # Source restriction level, 0-No additional restrictions; general terms of the license agreement apply, 9-General terms + SNOMED CT Affiliate License in Appendix 2
            SUPPRESS = fields[16] # Suppressible flag. O: All obsolete content, N: None of the above
            CVF = fields[17] # Content View Flag, Bit field used to flag rows included in Content View.

            find = False
            for document in train_data:
                for entity in document.entities:
                    if CODE in entity.norm_ids:
                        find = True
                        break

            if find:
                out.write(line)


    out.close()



if __name__=="__main__":
    pass
# UMLS_dict = load_umls_MRCONSO("/Users/feili/UMLS/2016AA/META/MRCONSO.RRF")
#
# h1 = UMLS_dict['C0000163']
# h2 = UMLS_dict['C0000727']
    make_debug_dict()
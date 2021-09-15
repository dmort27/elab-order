import re

HD_SYLLABLE = re.compile('''(?P<ons>f|v|xy|x|s|z|y|h|
                                n?(?:pl|tx|ts|p|t|r|c|k|q)h?|h?|d|dh|
                                  h?(?:ny|n|ml|m|l)|)
                            (?P<rhy>ee|oo|ai|aw|au|ia|ua|i|e|a|o|u|w)
                            (?P<ton>b|s|j|v|m|g|d|)''', flags=re.X)

ML_SYLLABLE = re.compile('''(?P<ons>f|v|xy|x|s|z|y|h|
                                n?(?:dl|pl|tx|ts|p|t|r|c|k|q)h?|
                                  (?:ny|n|ml|m|hl|l)|)
                            (?P<rhy>aa|ee|oo|ai|aw|au|ua|i|e|a|o|u|w)
                            (?P<ton>b|s|j|v|m|g|d|)''', flags=re.X)

RPA_SYLLABLE = re.compile('''(?P<ons>f|v|xy|x|s|z|y|h|
                                n?(?:dl|pl|tx|ts|p|t|r|c|k|q)h?|h?|d|dh|
                                  h?(?:ny|n|ml|m|l)|)
                            (?P<rhy>aa|ee|oo|ai|aw|au|ia|ua|i|e|a|o|u|w)
                            (?P<ton>b|s|j|v|m|g|d|)''', flags=re.X)
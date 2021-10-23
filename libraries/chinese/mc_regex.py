import re

LIRONG_SYLLABLE = re.compile('''(?P<ons>t͡sʰ?|d͡z|t͡ɕʰ?|d͡ʑ|t͡ʃʰ?|d͡ʒ|[bmdnȡȵlszʃɕʑgɡŋʔxɣ]|[ptȶk]ʰ?|)
                            (?P<rhy>[ɑaᴀɛəeĕɐijoɔu]*[mnŋ]?)
                            (?P<ton>[HX]|[ptk]̚|)''', flags=re.X)
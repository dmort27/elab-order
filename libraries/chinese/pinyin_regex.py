import re

# this regex will over match
# so the assumption is that the input pinyin is well formed
PINYIN_SYLLABLE = re.compile('''(?P<ons>[bpmfdtnlgkhjqxrwy]|[zcs]h?|)
                            (?P<rhy>[aeiou:]*n?g?|er)
                            (?P<ton>[12345])''', flags=re.X)


if __name__ == '__main__':
    test = ['wei2', 'bai4', 'fei1', 'nu:3', 'er3', 'lu:e3']
    for t in test:
        m = PINYIN_SYLLABLE.match(t)
        print(m.group('ons'), m.group('rhy'), m.group('ton'))
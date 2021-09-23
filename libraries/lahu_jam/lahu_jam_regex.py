import re

LAHU_REGEX = re.compile('''(?P<ons>[ptckq]h?|[bdjgmnŋfhvyl]|š|g̈)
                           (?P<rhy>[iɨueəoɛaɔ])
                           (?P<ton>\u0302ʔ|\u0300ʔ|\u0301|\u0302|\u0300|\u0304|)''',
                        flags=re.X)

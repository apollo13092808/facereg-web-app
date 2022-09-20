from unidecode import unidecode


def standardize(s=''):
    name = unidecode(s)
    name = name.strip().title()
    name = ' '.join([i for i in name.split()])
    return name

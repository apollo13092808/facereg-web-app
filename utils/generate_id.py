import random
import string


def generate(size=4):
    return ''.join(random.choice(string.digits) for n in range(size))

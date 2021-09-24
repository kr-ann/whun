import random
import string
import re

FILE = './datasets/dataset_QA'

with open(FILE,'r', encoding='utf-8') as source:
    data = [(random.random(), line) for line in source]  # randomize

data.sort()

with open(FILE + '_clean.txt', 'w', encoding='utf-8') as output:
    for _, line in data:
        if all(x in string.printable for x in line):  # delete lines with unprintable chars
            output.write(re.sub(' +', ' ', line))  # many spaces to one
            # output.write("Joke: " + re.sub(' +', ' ', line))

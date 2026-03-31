import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'itergen'))

from itergen.main import IterGen

with open(os.path.join(os.path.dirname(__file__), 'grammar.txt')) as f:
    grammar = f.read()


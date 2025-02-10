import os
import pandas as pd
import numpy as np
import spacy
import unidecode
import contractions as contract
import re
import wordninja
import collections
import pkg_resources
from spellchecker import SpellChecker
from symspellpy import SymSpell, Verbosity

class DP:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        print("Data Preprocessing is activated!")

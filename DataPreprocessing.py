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
        print("Data Preprocessing is activated!")

        self.dataframe = dataframe
        self.deselect_stop_words = ['no', 'not', 'a', 'the', 'so']

        # Defining methods
        self.nlp = spacy.load("en_core_web_sm")
        self.vocab = collections.Counter()
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self.dictionary_path = "Datasets/frequency_dictionary_en_82_765.txt"
        self.bigram_path = "Datasets/frequency_bigramdictionary_en_243_342.txt"
        self.sym_spell.load_dictionary(self.dictionary_path, term_index=0, count_index=1)
        self.sym_spell.load_bigram_dictionary(self.bigram_path, term_index=0, count_index=2)

        print("en_core_web_sm dictionary loaded successfuly!")

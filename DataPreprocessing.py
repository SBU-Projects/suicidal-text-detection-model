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

        #Defining methods
        self.nlp = spacy.load("en_core_web_sm")
        self.vocab = collections.Counter()
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self.dictionary_path = "Datasets/frequency_dictionary_en_82_765.txt"
        self.bigram_path = "Datasets/frequency_bigramdictionary_en_243_342.txt"
        self.sym_spell.load_dictionary(self.dictionary_path, term_index=0, count_index=1)
        self.sym_spell.load_bigram_dictionary(self.bigram_path, term_index=0, count_index=2)

        print("en_core_web_sm dictionary loaded successfuly!")

    # Spell Check using Symspell
    def fix_spelling(self, text):
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        correctedtext = suggestions[
            0].term  # get the first suggestion, otherwise returns original text if nothing is corrected
        return correctedtext

    # Remove some important words from stopwords list
    def deselect_stop_words(self):
        for w in self.deselect_stop_words:
            self.nlp.vocab[w].is_stop = False

    # Remove extra whitespaces from text
    def remove_whitespace(self, text):
        text = text.strip()
        return " ".join(text.split())

    def remove_accented_chars(self, text):
        text = unidecode.unidecode(text)
        return text

    # Remove URL
    def remove_url(self, text):
        return re.sub(r'http\S+', '', text)

    # Removing symbols and digits
    def remove_symbols_digits(self, text):
        return re.sub('[^a-zA-Z\s]', ' ', text)

    # Removing special characters
    def remove_special(self, text):
        return text.replace("\r", " ").replace("\n", " ").replace("    ", " ").replace('"', '')

    # Fix word lengthening (characters are wrongly repeated)
    def fix_lengthening(self, text):
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", text)

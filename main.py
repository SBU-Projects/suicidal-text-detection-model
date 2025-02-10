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

# Load dataset
df = pd.read_csv('Datasets/Suicide_Ideation_Dataset(Twitter-based).csv')
df.reset_index(drop=True, inplace=True)
""".
df['Suicide'] = df['Suicide'].replace({'Potential Suicide post ': 1, 'Not Suicide post': 0})
print(df)
."""
# Defining methods

nlp = spacy.load("en_core_web_sm")
vocab = collections.Counter()
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "Datasets/frequency_dictionary_en_82_765.txt"
bigram_path = "Datasets/frequency_bigramdictionary_en_243_342.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)


def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text

sample = 'café'
print(sample)
print(sample)
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


# Spell Check using Symspell
def fix_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    correctedtext = suggestions[
        0].term  # get the first suggestion, otherwise returns original text if nothing is corrected
    return correctedtext


# Remove some important words from stopwords list
deselect_stop_words = ['no', 'not', 'a', 'the', 'so']

for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False


# Remove extra whitespaces from text
def remove_whitespace(text):
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text

# Remove URL
def remove_url(text):
    return re.sub(r'http\S+', '', text)

# Removing symbols and digits
def remove_symbols_digits(text):
    return re.sub('[^a-zA-Z\s]', ' ', text)

# Removing special characters
def remove_special(text):
    return text.replace("\r", " ").replace("\n", " ").replace("    ", " ").replace('"', '')

# Fix word lengthening (characters are wrongly repeated)
def fix_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

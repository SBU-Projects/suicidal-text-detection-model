import random
from nltk.corpus import wordnet


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_sentence = words.copy()
    random_word_list = list(set([word for word in words if len(get_synonyms(word)) > 0]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) > 0:
            synonym = random.choice(synonyms)
            new_sentence = [synonym if word == random_word else word for word in new_sentence]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_sentence)


# مثال استفاده
sentence = "I love programming in Python"
similar_sentence = synonym_replacement(sentence, n=2)
print(similar_sentence)
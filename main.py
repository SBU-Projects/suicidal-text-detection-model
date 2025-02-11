import nlpaug.augmenter.word as naw

# ایجاد جملات مشابه با استفاده از WordNet
aug = naw.SynonymAug(aug_src='wordnet')
sentence = "I love programming in Python"
similar_sentence = aug.augment(sentence)
print(similar_sentence)
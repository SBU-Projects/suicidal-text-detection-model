import random

def random_insertion(sentence, n=1):
    words = sentence.split()
    for _ in range(n):
        word = random.choice(words)
        words.insert(random.randint(0, len(words)), word)
    return ' '.join(words)

def random_deletion(sentence, p=0.1):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    remaining = list(filter(lambda x: random.random() > p, words))
    if len(remaining) == 0:
        return random.choice(words)
    return ' '.join(remaining)

# مثال استفاده
sentence = "I want to kill myself"
similar_sentence = random_insertion(sentence, n=2)
print(similar_sentence)
similar_sentence = random_deletion(sentence, p=0.2)
print(similar_sentence)
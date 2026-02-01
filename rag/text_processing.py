import nltk

def only_worlds(tokens):
    words = []
    for token in tokens:
        if(token.isalpha()):
            words.append(token)

    return words

def remove_stopwords(tokens):
    filtered_words = []
    stop_words_pt = set(nltk.corpus.stopwords.words('portuguese'))
    stop_words_en = set(nltk.corpus.stopwords.words('english'))

    stop_words = stop_words_pt.union(stop_words_en)
    for token in tokens:
        if(not token in stop_words):
            filtered_words.append(token)

    return filtered_words

def text_processing(text):
    tokens = nltk.word_tokenize(text)
    words = only_worlds(tokens)
    lower_tokens = []
    for token in words:
        lower_tokens.append(token.lower())

    filtered_words = remove_stopwords(lower_tokens)

    return filtered_words
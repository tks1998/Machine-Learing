import re
import contractions
import unicodedata
import inflect
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


def remove_punctuation(text, remove_digit=False):
    """xóa dấu câu (và số nếu muốn)"""
    if remove_digit:
        pattern = r'[^a-zA-z\s]'
    else:
        pattern = r'[^a-zA-z0-9\s]'

    text = re.sub(pattern, '', text)
    return text


def remove_non_ascii(words):
    """xóa các kí tự non_ASCII"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """chuyển số nguyên từ dạng số sang dạng bằng chữ"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """xóa stopwords"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def preprocessing(terms, remove_non_ACSII=True, replace_num=True, lemmatize=True, stemming=True):
    if remove_non_ACSII:
        terms = remove_non_ascii(terms)
    if replace_num:
        terms = replace_numbers(terms)
    terms = remove_stopwords(terms)
    if lemmatize:
        terms = lemmatize_verbs(terms)
    if stemming:
        terms = stem_words(terms)

    return terms


def get_terms(text, expand_contraction=True, remove_punc=True, remove_digit=False, remove_non_ACSII=True, replace_num=True, lemmatize=True, stemming=True):
    text = text.lower()
    if expand_contraction:
        """ mở rộng các từ viết tắt """
        text = contractions.fix(text)
    if remove_punc:
        text = remove_punctuation(text, remove_digit=remove_digit)
    terms = nltk.word_tokenize(text)
    terms = preprocessing(terms, remove_non_ACSII=remove_non_ACSII,
                          replace_num=replace_num, lemmatize=lemmatize, stemming=stemming)
    return terms

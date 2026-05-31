"""Text preprocessing cleaners for NLP classification pipelines.

Provides two cleaner classes — StemCleaner (Porter stemming via NLTK) and
LemmaCleaner (lemmatisation via spaCy) — each applying a configurable chain of
normalisation, tokenisation, and noise-removal steps suitable for short,
informal customer-service text.
"""

import re
import spacy
from string import punctuation

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Domain-specific slang/typo mapping for customer-service social media text
NORMALISATION_MAP = {
    'acct': 'account',
    'accts': 'accounts',
    'pls': 'please',
    'plz': 'please',
    'pwd': 'password',
    'wout': 'without',
    'emial': 'email',
    'wat': 'what',
    '2': 'to',
    'cant': 'cannot',
    'didnt': 'did not',
    'doesnt': 'does not',
    'dont': 'do not',
    'wont': 'will not',
    'isnt': 'is not',
}

class StemCleaner:
    """NLTK-based preprocessing pipeline using Porter stemming.

    Applies a sequential chain: lowercase -> punctuation removal ->
    [normalisation] -> tokenisation -> [numeric removal] -> [stop word removal]
    -> stemming -> rejoin. Optional steps are controlled by boolean flags.

    Args:
        data: List of raw text strings to preprocess.
        norm: Apply domain-specific slang/typo normalisation.
        num: Remove standalone numeric tokens.
        stop: Remove stop words (negations are always retained).
    """

    NORMALISATION_MAP = NORMALISATION_MAP

    def __init__(self, data, norm=False, num=False, stop=False):
        self.data = data
        self.negations = {'not', 'no', 'cannot'}
        # Preserve negations to retain sentiment signals in classification
        self.stop_words = set(stopwords.words('english')) - self.negations

        self.lowered = self.lowercase(self.data)
        self.punctuated = self.remove_punctuation(self.lowered)
        self.normalised = self.normalise_text(self.punctuated) if norm else self.punctuated
        self.tokenised = self.tokenise(self.normalised)
        self.numerics_removed = self.remove_standalone_numerics(self.tokenised) if num else self.tokenised
        self.cleaned = self.remove_stop_words(self.numerics_removed) if stop else self.numerics_removed
        self.stemmed = self.stem(self.cleaned)
        self.rejoined = self.rejoin(self.stemmed)

    def lowercase(self, texts):
        """Convert all text to lowercase."""
        return [t.lower() for t in texts]

    def remove_punctuation(self, texts):
        """Strip ASCII and common Unicode punctuation (em-dash, curly quotes, ellipsis)."""
        extra = '\u2014\u2013\u201c\u201d\u2018\u2019\u2026'
        all_punct = punctuation + extra
        cleaned_texts = []
        for t in texts:
            t = t.translate(str.maketrans('', '', all_punct))
            cleaned_texts.append(t)
        return cleaned_texts

    def normalise_text(self, texts):
        """Replace informal shorthand with formal equivalents using word-boundary matching."""
        normalised = []
        for t in texts:
            for informal, formal in self.NORMALISATION_MAP.items():
                t = re.sub(r'\b' + re.escape(informal) + r'\b', formal, t)
            normalised.append(t)
        return normalised

    def tokenise(self, texts):
        """Tokenise using NLTK's word_tokenize (Penn Treebank style)."""
        return [word_tokenize(t) for t in texts]

    def remove_standalone_numerics(self, tokenised_texts):
        """Remove tokens that are purely numeric (e.g. '123') but keep alphanumeric."""
        cleaned = []
        for t in tokenised_texts:
            cleaned.append([w for w in t if not re.fullmatch(r'\d+', w)])
        return cleaned

    def remove_stop_words(self, tokenised_texts):
        """Remove English stop words, retaining negations for sentiment preservation."""
        cleaned_texts = []
        for t in tokenised_texts:
            cleaned_texts.append([w for w in t if w not in self.stop_words])
        return cleaned_texts

    def stem(self, cleaned_texts):
        """Apply Porter stemming to reduce words to their root form."""
        porter = PorterStemmer()
        stemmed_texts = []
        for t in cleaned_texts:
            stemmed_texts.append([porter.stem(w) for w in t])
        return stemmed_texts

    def rejoin(self, stemmed_texts):
        """Rejoin token lists into space-separated strings for vectorisation."""
        return [' '.join(tokens) for tokens in stemmed_texts]
    
class LemmaCleaner:
    """spaCy-based preprocessing pipeline using lemmatisation.

    Same configurable chain as StemCleaner but uses spaCy's lemmatiser
    (en_core_web_sm) instead of Porter stemming, producing more linguistically
    accurate root forms at the cost of higher compute.

    Args:
        data: List of raw text strings to preprocess.
        norm: Apply domain-specific slang/typo normalisation.
        num: Remove standalone numeric tokens.
        stop: Remove stop words (negations are always retained).
    """

    NORMALISATION_MAP = NORMALISATION_MAP

    def __init__(self, data, norm=False, num=False, stop=False):
        self.data = data
        self.nlp = spacy.load("en_core_web_sm")
        self.negations = {'not', 'no', 'cannot'}
        # Preserve negations to retain sentiment signals in classification
        self.stop_words = set(self.nlp.Defaults.stop_words) - self.negations

        self.lowered = self.lowercase(self.data)
        self.punctuated = self.remove_punctuation(self.lowered)
        self.normalised = self.normalise_text(self.punctuated) if norm else self.punctuated
        self.tokenised = self.tokenise(self.normalised)
        self.numerics_removed = self.remove_standalone_numerics(self.tokenised) if num else self.tokenised
        self.stop_words_removed = self.remove_stop_words(self.numerics_removed) if stop else self.numerics_removed
        self.lemmatised = self.lemmatise(self.stop_words_removed)
        self.rejoined = self.rejoin(self.lemmatised)

    def lowercase(self, texts):
        """Convert all text to lowercase."""
        return [t.lower() for t in texts]

    def remove_punctuation(self, texts):
        """Strip ASCII and common Unicode punctuation (em-dash, curly quotes, ellipsis)."""
        extra = '—–""''…'
        all_punct = punctuation + extra
        cleaned_texts = []
        for t in texts:
            t = t.translate(str.maketrans('', '', all_punct))
            cleaned_texts.append(t)
        return cleaned_texts

    def normalise_text(self, texts):
        """Replace informal shorthand with formal equivalents using word-boundary matching."""
        normalised = []
        for t in texts:
            for informal, formal in self.NORMALISATION_MAP.items():
                t = re.sub(r'\b' + re.escape(informal) + r'\b', formal, t)
            normalised.append(t)
        return normalised

    def tokenise(self, texts):
        """Tokenise using spaCy's pipeline, filtering whitespace-only tokens."""
        result = []
        for t in texts:
            doc = self.nlp(t)
            result.append([token for token in doc if token.text.strip()])
        return result

    def remove_standalone_numerics(self, tokenised_texts):
        """Remove tokens that are purely numeric (e.g. '123') but keep alphanumeric."""
        cleaned = []
        for t in tokenised_texts:
            cleaned.append([tok for tok in t if not re.fullmatch(r'\d+', tok.text)])
        return cleaned

    def remove_stop_words(self, tokenised_texts):
        """Remove English stop words, retaining negations for sentiment preservation."""
        cleaned_texts = []
        for t in tokenised_texts:
            cleaned_texts.append([tok for tok in t if tok.text not in self.stop_words])
        return cleaned_texts

    def lemmatise(self, tokenised_texts):
        """Reduce tokens to lemmas via spaCy's morphological analysis."""
        result = []
        for tokens in tokenised_texts:
            result.append([tok.lemma_ for tok in tokens])
        return result

    def rejoin(self, lemmatised_texts):
        """Rejoin token lists into space-separated strings for vectorisation."""
        return [' '.join(tokens) for tokens in lemmatised_texts]
import re
import spacy
from string import punctuation

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

class StemCleaner:  
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
    
    def __init__(self, data, norm=False, num=False, stop=False):
        self.data = data
        self.negations = {'not', 'no', 'cannot'}
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
        return [t.lower() for t in texts]
    
    def remove_punctuation(self, texts):
        extra = '\u2014\u2013\u201c\u201d\u2018\u2019\u2026'  
        all_punct = punctuation + extra
        cleaned_texts = []
        for t in texts:
            t = t.translate(str.maketrans('', '', all_punct))
            cleaned_texts.append(t)
        return cleaned_texts
    
    def normalise_text(self, texts):
        normalised = []
        for t in texts:
            for informal, formal in self.NORMALISATION_MAP.items():
                t = re.sub(r'\b' + re.escape(informal) + r'\b', formal, t)
            normalised.append(t)
        return normalised
    
    def tokenise(self, texts):
        return [word_tokenize(t) for t in texts]
    
    def remove_standalone_numerics(self, tokenised_texts):
        cleaned = []
        for t in tokenised_texts:
            cleaned.append([w for w in t if not re.fullmatch(r'\d+', w)])
        return cleaned
    
    def remove_stop_words(self, tokenised_texts):
        cleaned_texts = []
        for t in tokenised_texts:
            cleaned_texts.append([w for w in t if w not in self.stop_words])
        return cleaned_texts
    
    def stem(self, cleaned_texts):
        porter = PorterStemmer()
        stemmed_texts = []
        for t in cleaned_texts:
            stemmed_texts.append([porter.stem(w) for w in t])
        return stemmed_texts
    
    def rejoin(self, stemmed_texts):
        return [' '.join(tokens) for tokens in stemmed_texts]
    
class LemmaCleaner:  
    NORMALISATION_MAP = {
        'acct': 'account',    # ~45 occurrences
        'accts': 'accounts',  # 1 occurrence
        'pls': 'please',      # ~12 occurrences
        'plz': 'please',
        'pwd': 'password',    # 1 occurrence
        'wout': 'without',    # w/out after punctuation removal
        'emial': 'email',     # 8 occurrences
        'wat': 'what',        # ~38 occurrences
        '2': 'to',            # ~30 occurrences
        'cant': 'cannot',     # 7 occurrences — expands so negation is preserved
        'didnt': 'did not',   # 1 occurrence
        'doesnt': 'does not',
        'dont': 'do not',
        'wont': 'will not',
        'isnt': 'is not',
    }

    def __init__(self, data, norm=False, num=False, stop=False):
        self.data = data
        self.nlp = spacy.load("en_core_web_sm")
        self.negations = {'not', 'no', 'cannot'}
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
        return [t.lower() for t in texts]
    
    def remove_punctuation(self, texts):
        extra = '—–""''…' 
        all_punct = punctuation + extra
        cleaned_texts = []
        for t in texts:
            t = t.translate(str.maketrans('', '', all_punct))
            cleaned_texts.append(t)
        return cleaned_texts
    
    def normalise_text(self, texts):
        normalised = []
        for t in texts:
            for informal, formal in self.NORMALISATION_MAP.items():
                t = re.sub(r'\b' + re.escape(informal) + r'\b', formal, t)
            normalised.append(t)
        return normalised
    
    def remove_stop_words(self, tokenised_texts):
        cleaned_texts = []
        for t in tokenised_texts:
            cleaned_texts.append([w for w in t if w not in self.stop_words])
        return cleaned_texts
    
    def tokenise(self, texts):
        result = []
        for t in texts:
            doc = self.nlp(t)
            tokens = [token.text for token in doc if token.text.strip()]
            result.append(tokens)
        return result

    def remove_standalone_numerics(self, tokenised_texts):
        cleaned = []
        for t in tokenised_texts:
            cleaned.append([w for w in t if not re.fullmatch(r'\d+', w)])
        return cleaned

    def lemmatise(self, tokenised_texts):
        result = []
        for tokens in tokenised_texts:
            doc = self.nlp(' '.join(tokens))
            result.append([
                token.lemma_ for token in doc
                if token.text.strip()
            ])
        return result
    
    def rejoin(self, lemmatised_texts):
        return [' '.join(tokens) for tokens in lemmatised_texts]
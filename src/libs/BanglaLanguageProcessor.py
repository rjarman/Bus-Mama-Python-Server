from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
import pandas as pd

from bangla import Words, Lemmatizer
from Wrappers import clean_check
from utils.config import Config

import os
import pickle

class Preprocessor:
    def __init__(self, context_manager):
        self.__stop_words = None
        self.__punctuation = None

        self.__context_manager = context_manager

    @clean_check
    def clean(self, sentence, stop_words = None, punctuation = None):
        """
        parameters:
            sentence: (str, list)
            stop_words: (list, None) of words to remove, default is None
            punctuation: (list, None) of punctuation marks to remove, default is None
        return:
            (iterator) str of every words in sentence
        """
        self.__stop_words = stop_words 
        self.__punctuation = punctuation
        temp_sentence = sentence
        if self.__stop_words: 
            for i in self.__stop_words: temp_sentence = temp_sentence.replace(i, '')
        if self.__punctuation: 
            for i in self.__punctuation: temp_sentence = temp_sentence.replace(i, '')

        return ' '.join(temp_sentence.split())
    
    def lemmatize(self, words):
        """
        parameters:
            words: (str or list) takes str or list of words
        return:
            (iterator) str of lemma word  
        """
        yield from self.__context_manager.lemmatizer.get_lemma(words)

    def splitter(self):
        """
        return:
            (feature_train, feature_test, label_train, label_test)
        """
        return train_test_split(
            self.__context_manager.WORDS_DATA,
            self.__context_manager.TAGS_DATA,
            test_size = self.__context_manager.test_size,
            random_state = self.__context_manager.random_state
        )
    
    def tfidf_vectorizer(self, feature_train, feature_test):
        """
        parameter:
            feature_train: (iterable) An iterable which yields either str, unicode or file objects
            feature_test: (iterable) An iterable which yields either str, unicode or file objects.
        return:
            sparse matrix of (n_samples, n_features), sparse matrix of (n_samples, n_features)
        """
        vectorizer = TfidfVectorizer(
            max_df=self.__context_manager.max_df,
            min_df=self.__context_manager.min_df,
            max_features=self.__context_manager.max_features,
            norm=self.__context_manager.norm,
            smooth_idf=self.__context_manager.smooth_idf,
            sublinear_tf=self.__context_manager.sublinear_tf
        )
        feature_train, feature_test = vectorizer.fit_transform(feature_train), vectorizer.transform(feature_test)

        return feature_train, feature_test

        # return vector, pd.DataFrame(vector.todense(), columns=vectorizer.get_feature_names())
    
    def select_percentile(self, feature_train, label_train, feature_test):
        """
        parameter:
            feature_train: array of shape [n_samples, n_features]
            feature_test: array of shape [n_samples, n_features]
        return:
            array of shape [n_samples, n_selected_features], array of shape [n_samples, n_selected_features]
        """
        selector = SelectPercentile(percentile=self.__context_manager.percentile)
        selector.fit(feature_train, label_train)
        feature_train, feature_test = selector.transform(feature_train).toarray(), selector.transform(feature_test).toarray()

        return feature_train, feature_test

class ContextManager:
    def __init__(self):
        # splitter()
        self.test_size, self.random_state = Config.SPLITTER['test_size'], Config.SPLITTER['random_state']
        # tfidf_vectorizer()
        self.max_df = Config.TFIDF['max_df']
        self.min_df = Config.TFIDF['min_df']
        self.max_features = Config.TFIDF['max_features']
        self.norm = Config.TFIDF['norm']
        self.smooth_idf = Config.TFIDF['smooth_idf']
        self.sublinear_tf = Config.TFIDF['sublinear_tf']
        # select_percentile()
        self.percentile = Config.PERCENTILE['percentile']
        # ContextManager
        self.__QUESTION_ANS, self.__QUESTION_DOMAIN = Config.PATH['question_ans'], Config.PATH['question_domain']
        
        # handling files data
        self.WORDS_DATA = None
        self.TAGS_DATA = None

        self.__file_handler()

        # handling lemmatizer
        self.lemmatizer = Lemmatizer()

    def __file_handler(self):
        try:
            if not (os.path.exists(self.__WORDS_PATH) and os.path.exists(self.__TAGS_PATH)): raise FileExistsError
        except FileExistsError:
            print('FileExistsError: missing data files in ./data/')
        else:
            with open(self.__WORDS_PATH, mode='rb') as file:
                self.WORDS_DATA = pickle.loads(file, encoding='utf-8')
            with open(self.__TAGS_PATH, mode='rb') as file:
                self.TAGS_DATA = pickle.loads(file, encoding='utf-8')

    

if __name__ == '__main__':
    words = Words()

    context_manager = ContextManager()

    preprocessor = Preprocessor(context_manager)
    print("""djujhd kdfgwked sdkfjh! djilkd.        
    
    dhd""")
    cleaned_word = preprocessor.clean("""djujhd kdfgwked sdkfjh! djilkd.        
    
    dhd""", stop_words=['kdfgwked'], punctuation=words.PUNCTUATION )
    print(list(cleaned_word))

    documentA = 'the man went out for a walk'
    documentB = 'the children sat around the fire'
    # preprocessor.vectorizer([documentA, documentB])
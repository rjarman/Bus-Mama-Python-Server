from .Wrappers import clean_check

class BLP:
    def __init__(self):
        pass

    @clean_check
    def clean(self, sentence, stop_words = None, punctuation = None):
        """
        parameters:
            sentence: (str)
            stop_words: (list) of words to remove, default is None
            punctuation: (list) of punctuation marks to remove, default is None
        return:
            (iterator) str of every words in sentence
        """
        yield 'a'
        # sentence = ()
        # rmv_punc = (word for i in punctuation if punctuation)
        # rmv_stop_words = (rmv_punc.replace(i, '') for i in stop_words if stop_words)
        # yield list()
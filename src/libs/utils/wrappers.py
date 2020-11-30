def clean_check(function):
    validation = [
        ('sentence', list),
        ('sentence', str),
        ('stop_words', list),
        ('stop_words', None),
        ('punctuation', list),
        ('punctuation', None)
    ]
    def wrapper(*args, **kwargs):
        sentence = args[1] if 'sentence' not in kwargs.items() else kwargs['sentence']
        for i in kwargs.items():
            if not isinstance(i, str): continue
            if (i, type(kwargs[i])) not in validation: raise KeyError(f'Check parameters!')

        if len(args) >= 2 and not (isinstance(sentence, str) or isinstance(sentence, list)): raise KeyError(f'Check parameters!')
        
        if isinstance(sentence, list):
            for i in sentence:
                yield function(*(args[0], i), **kwargs)
        else:
            yield function(*args, **kwargs)
    return wrapper
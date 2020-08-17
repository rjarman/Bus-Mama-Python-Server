def clean_check(function):
    validation = ((list, None), ('stop_words', 'punctuation'))
    def wrapper(*args, **kwargs):
        for i in kwargs:
            if i == 'sentence' and not isinstance(kwargs['sentence'], str): raise KeyError(f'Check parameters!')
            if i in validation[1] and (type(kwargs['stop_words']), type(kwargs['punctuation'])) not in validation: raise KeyError(f'Check parameters!')

        if len(args) == 1: pass
        elif len(args) == 2 and not isinstance(args[1], str): raise KeyError(f'Check parameters!')
        else: raise KeyError(f'Check parameters!')
        
        yield from function(*args)
    return wrapper
class ColorList:
    __HEADER = '\033[95m'
    __BLUE = '\033[94m'
    __CYAN = '\033[96m'
    __GREEN = '\033[92m'
    __WARNING = '\033[93m'
    __FAIL = '\033[91m'
    __ENDCOLOR = '\033[0m'
    __BOLD = '\033[1m'
    __UNDERLINE = '\033[4m'

    @classmethod
    def header(cls, data):
        return f'{ColorList.__HEADER}{data}{ColorList.__ENDCOLOR}'

    @classmethod
    def blue(cls, data):
        return f'{ColorList.__BLUE}{data}{ColorList.__ENDCOLOR}'

    @classmethod
    def cyan(cls, data):
        return f'{ColorList.__CYAN}{data}{ColorList.__ENDCOLOR}'

    @classmethod
    def green(cls, data):
        return f'{ColorList.__GREEN}{data}{ColorList.__ENDCOLOR}'

    @classmethod
    def warning(cls, data):
        return f'{ColorList.__WARNING}{data}{ColorList.__ENDCOLOR}'

    @classmethod
    def fail(cls, data):
        return f'{ColorList.__FAIL}{data}{ColorList.__ENDCOLOR}'

    @classmethod
    def bold(cls, data):
        return f'{ColorList.__BOLD}{data}{ColorList.__ENDCOLOR}'

    @classmethod
    def underline(cls, data):
        return f'{ColorList.__UNDERLINE}{data}{ColorList.__ENDCOLOR}'

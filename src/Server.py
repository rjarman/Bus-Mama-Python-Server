from libs.BanglaLanguageProcessor import BLP

class Server(BLP):

    def __init__(self):
        BLP.__init__(self)
        blp = BLP()
        blp.clean('jyaghk')

if __name__ == '__main__':
    server = Server()
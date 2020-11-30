from bangla import Files
from config import Config
import pickle
import pandas as pd

class Data:
    def __init__(self):
        self.__files = Files()
    
    def add_data(self, d_type, words):
        self.__files.add_data(d_type, words)
        
    def pickle_me(self, _from, _to):
        # pickle.dump(data, open(self.__path, mode='wb'))
        # pd.DataFrame(pd.read_excel(_from)).to_pickle(_to)
        with open(_from, mode='r') as file: 
            print(file.readlines())
    
    def xlsx_me(self, _from, _to):
        data = pickle.load(open(_from, mode='rb'))
        pd.DataFrame(data).to_excel(_to)
        
if __name__ == '__main__':
    data = Data()
    data.xlsx_me( Config.PATH['question_ans'], Config.PATH['original_questions_ans'])
    data.xlsx_me( Config.PATH['question_domain'], Config.PATH['original_questions_domains'])
    # data.add_data('words_dict', 'এ')
    # data.pickle_me(Config.PATH['original_questions_ans_csv'], Config.PATH['question_ans'])
    # data.pickle_me(Config.PATH['original_questions_domains_csv'], Config.PATH['question_domain'])
    
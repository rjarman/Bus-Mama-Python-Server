import os


class Config:
    __dir = os.path.dirname(__file__)

    SERVER = {
        'host': 'localhost',
        'port': 8080
    }

    PATH = {
        'question_ans': __dir + '/../../data/bangla_questions_ans.pkl',
        'question_domain': __dir + '/../../data/bangla_questions_domain.pkl',
        'original_questions_ans': __dir + '/../../data/original_questions_ans.xlsx',
        'original_questions_domains': __dir + '/../../data/original_questions_domains.xlsx',
        'original_questions_ans_csv': __dir + '/../../data/original_questions_ans.csv',
        'original_questions_domains_csv': __dir + '/../../data/original_questions_domains.csv',
        'log': __dir + '/../../logs/server_log.csv'
    }

    SPLITTER = {
        'test_size': 0.3,
        'random_state': 42
    }

    TFIDF = {
        'max_df': 1,
        'min_df': 1,
        'max_features': None,
        'norm': 'l2',
        'smooth_idf': True,
        'sublinear_tf': True
    }

    PERCENTILE = {
        'score_func': 'f_classif',
        'percentile': 10
    }

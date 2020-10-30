import os


class Config:

    SERVER = {
        'host': 'localhost',
        'port': 8080
    }

    PATH = {
        'question_ans': os.path.dirname(__file__) + '/../../data/bangla_questions_ans.pkl',
        'question_domain': os.path.dirname(__file__) + '/../../data/bangla_questions_domain.pkl',
        'log': os.path.dirname(__file__) + '/../../logs/server_log.csv'
    }

    SPLITTER = {
        'test_size': 0.1,
        'random_state': 42
    }

    TFIDF = {
        'max_df': 0.5,
        'min_df': 0.5,
        'max_features': None,
        'norm': 'l2',
        'smooth_idf': True,
        'sublinear_tf': True
    }

    PERCENTILE = {
        'score_func': 'f_classif',
        'percentile': 10
    }

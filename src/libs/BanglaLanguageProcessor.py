from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
import pandas as pd

from bangla import Words, Lemmatizer, Files
from .utils.wrappers import clean_check
from .utils.config import Config

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
        sensitive_words = ['কখনো']
        test_stop = ['আপনি', 'আপনার', 'সাধারণত', 'এই', 'তুমি', 'একজন', 'একটি', 'শুধু', 'আমি', 'আমার', 'অবশ্যই', 'কি', 'কী', 'কখন']
        
        if test_stop or sensitive_words:
            unique_set = set(temp_sentence.split(' '))
            temp_sentence = ' '.join(unique_set)
            for i in sensitive_words: 
                temp_sentence = temp_sentence.replace(i, '')
            for i in test_stop: 
                temp_sentence = temp_sentence.replace(i, '')
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

    def splitter(self, X, Y):
        """
        return:
            (feature_train, feature_test, label_train, label_test)
        """
        return train_test_split(
            X,
            Y,
            test_size = self.__context_manager.test_size,
            random_state = self.__context_manager.random_state
        )
    
    def tfidf_vectorizer(self, feature_train, feature_test):
        """
        parameter:
            feature_train: (iterable) An iterable which yields either str, unicode or file objects
            feature_test: (iterable) An iterable which yields either str, unicode or file objects.
        return:
            TfidfVectorizer object and sparse matrix of (n_samples, n_features), sparse matrix of (n_samples, n_features)
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

        return vectorizer, feature_train, feature_test

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

    def all(self, data, punctuation=None):
        try:
            processed_data = (lambda _d: [' '.join(list(self.lemmatize(i.split(' ')))) for j in list(_d.keys())
             for i in self.clean(_d[j].replace('\u200c', ''), punctuation=punctuation) if i != ''])
            questions_ans = []
            # print(list(self.clean('বশেমুরবিপ্রবি থেকে কাশিয়ানী'.replace('\u200c', ''), punctuation=punctuation)))
            # print(' '.join(list(self.lemmatize('কাশিয়ানী'.split(' ')))))
            for i in data:
                temp = {}
                _processed_data = processed_data(i)
                if len(i.keys()) == len(_processed_data):
                    for j, k in zip(_processed_data, i):
                        temp[k] = j
                    questions_ans.append(temp)
            return pd.DataFrame(questions_ans)
        except:
            print('Error: check parameters!')
        # try:
        #     processed_data = [' '.join(list(preprocessor.lemmatize(i.split(' ')))) for j in list(data.keys())
        #      for i in preprocessor.clean(data[j].replace('\u200c', ''), punctuation=words.PUNCTUATION)]
        #     return processed_data;
        # except:
        #     print(data)

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
        self.QUESTION_ANS = None
        self.QUESTION_DOMAIN = None
        self.__file_handler()

        # handling lemmatizer
        self.lemmatizer = Lemmatizer()
        self.ready_predictor()
        

    def __file_handler(self):
        try:
            if not (os.path.exists(self.__QUESTION_ANS) and os.path.exists(self.__QUESTION_DOMAIN)): raise FileExistsError
        except FileExistsError:
            print('FileExistsError: missing data files in ./data/')
        else:
            with open(self.__QUESTION_ANS, mode='rb') as file:
                self.QUESTION_ANS = pickle.load(file, encoding='utf-8')
            with open(self.__QUESTION_DOMAIN, mode='rb') as file:
                self.QUESTION_DOMAIN = pickle.load(file, encoding='utf-8')
    
    def ready_predictor(self):
        self.words = Words()
        self.preprocessor = Preprocessor(self)
    
        total_words = len(list(self.words.get_words()))
        print('total number of bangla words: ', total_words)
        
        updated_questions_ans = self.preprocessor.all(self.QUESTION_ANS, self.words.PUNCTUATION)
        updated_questions_domains = self.preprocessor.all(self.QUESTION_DOMAIN, self.words.PUNCTUATION)
       
        from sklearn.naive_bayes import MultinomialNB
        from sklearn import model_selection, naive_bayes, svm	
        from sklearn.preprocessing import LabelEncoder
        
        test = [
                ['আপনি কখন ক্যাম্পাস ছেড়েছেন?', 'জিপিএসটাইম', 'বাসচ্যাট'], 
                ['ক্যাম্পাস কি ছাড়বা?', 'জিপিএসটাইম', 'বাসচ্যাট'], 
                ['আপনার এখানে আসতে কতক্ষণ সময় লাগবে?', 'জিপিএসটাইম', 'বাসচ্যাট'], 
                ['কতক্ষণ লাগবে?', 'জিপিএসটাইম', 'বাসচ্যাট'],
                ['সাধারণত আপনি ক্যাম্পাস থেকে কখন বের হন?', 'সাধারণত সকাল ৮ টার দিকে বের হই।', 'বাসচ্যাট'], 
                ['কখন বের হন?', 'সাধারণত সকাল ৮ টার দিকে বের হই।', 'বাসচ্যাট'], 
                ['বের হও কখন?', 'সাধারণত সকাল ৮ টার দিকে বের হই।', 'বাসচ্যাট'], 
                ['আপনি এখন কোথায়?', 'জিপিএসলোকেশন', 'বাসচ্যাট'], 
                ['এই মুহূর্তে আপনার অবস্থান কোথায়?', 'জিপিএসলোকেশন', 'বাসচ্যাট'],
                ['এখন কোথায় আছো?', 'জিপিএসলোকেশন', 'বাসচ্যাট'], 
                ['আপনি এখান থেকে কত দূরে?', 'জিপিএসডিস্টেন্স', 'বাসচ্যাট'], 
                ['আপনার বর্তমান অবস্থান থেকে এখানে আসতে আপনাকে কতক্ষণ সময় নিতে হবে?', 'জিপিএসটাইম', 'বাসচ্যাট'] , 
                ['এখানে আসতে কতক্ষণ সময় লাগবে?', 'জিপিএসটাইম', 'বাসচ্যাট'], 
                ['মাম্মা কতক্ষণ লইবা?', 'জিপিএসটাইম', 'বাসচ্যাট'], 
                ['মামা কতক্ষণ লইবা?', 'জিপিএসটাইম', 'বাসচ্যাট'],
                ['লঞ্চঘাট যাবা', '#০০১', 'বাসচ্যাট'],
                ['পুলিশলাইন্স যাবা', '#০০২ ', 'বাসচ্যাট'],
                ['কাশিয়ানী  যাবা', '#০০৩', 	'বাসচ্যাট'],
                ['কাশিয়ানী  যাবেন?', '#০০৩', 	'বাসচ্যাট'],
                ['টুঙ্গিপাড়া যাবা',	'#০০৪', 'বাসচ্যাট'],
                ['টুঙ্গিপাড়া  যাবেন?',	'#০০৪', 'বাসচ্যাট'],
                ['বাগেরহাট যাবা', '#০০৫', 	'বাসচ্যাট'],
                ['খুলনা যাবা', '#০০৬',	'বাসচ্যাট'],
                ['বশেমুরবিপ্রবি','বশেমুরবিপ্রবিতে আপনাকে স্বাগতম!',	'বাসচ্যাট'],
                ['লঞ্চঘাট',	'#০০১', 'বাসচ্যাট'],
                ['পুলিশলাইন্স', '#০০২ ', 	'বাসচ্যাট'],
                ['কাশিয়ানী', '#০০৩', 'বাসচ্যাট'],
                ['টুঙ্গিপাড়া', 	'#০০৪','বাসচ্যাট'],
                ['বাগেরহাট',	'#০০৫', 'বাসচ্যাট'],
                ['খুলনা',	'#০০৬', 'বাসচ্যাট'],
               [ 'লঞ্চঘাট যাবে কোন  বাস?',	'#০০১', 'বাসচ্যাট'],
                ['পুলিশলাইন্স যাবে কোন  বাস?',	'#০০২', 'বাসচ্যাট'],
                ['কাশিয়ানী যাবে কোন  বাস?'	, '#০০৩' ,'বাসচ্যাট'],
                ['টুঙ্গিপাড়া যাবে কোন  বাস?'	, '#০০৪' , 'বাসচ্যাট'],
                ['বাগেরহাট যাবে কোন  বাস?', '#০০৫'	,'বাসচ্যাট'],
                ['খুলনা যাবে কোন  বাস?'	, '#০০৬','বাসচ্যাট']]
        test_df = pd.DataFrame(test, columns=['question', 'ans', 'tag'])	

        self.vect_reply = TfidfVectorizer()
        train_reply = self.vect_reply.fit_transform(test_df['question'])	
        self.encoder_reply = LabelEncoder()
        Train_Y_reply = self.encoder_reply.fit_transform(test_df['ans'])
        print(Train_Y_reply)
        print(self.encoder_reply.inverse_transform(Train_Y_reply))
        self.SVM_reply = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='scale', probability=True)
        self.SVM_reply.fit(train_reply,Train_Y_reply)
        
        self.vect_tags = TfidfVectorizer()
        train_tags = self.vect_tags.fit_transform(updated_questions_domains['sentences'])	
        self.encoder_tags = LabelEncoder()
        Train_Y_tags = self.encoder_tags.fit_transform(updated_questions_domains['tags'])
        self.SVM_tags = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='scale')
        self.SVM_tags.fit(train_tags,Train_Y_tags)
        
                
    def make_reply(self, recieved_message):
        clean = list(self.preprocessor.clean(recieved_message['message'], punctuation=self.words.PUNCTUATION))
        lemmatized = ' '.join(list(self.preprocessor.lemmatize(clean[0].split(' '))))
        
        predictions_SVM_reply = self.SVM_reply.predict(self.vect_reply.transform([lemmatized]))
        reply = self.encoder_reply.inverse_transform(predictions_SVM_reply)
        print(dict(zip(self.SVM_reply.classes_, self.SVM_reply.predict_proba(self.vect_reply.transform([lemmatized]))[0])))
        
        predictions_SVM_tags = self.SVM_tags.predict(self.vect_tags.transform([lemmatized]))
        tag = self.encoder_tags.inverse_transform(predictions_SVM_tags)
        if tag[0] != 'বাস': 
            return 'বুঝতে পারি নাই মামা, আবার কইবা ?', tag[0]
        return reply[0], tag[0]
    

if __name__ == '__main__':
    words = Words()
    files = Files()

    total_words = len(list(words.get_words()))
        
    context_manager = ContextManager()
    preprocessor = Preprocessor(context_manager)

    original_questions_ans = pd.DataFrame(context_manager.QUESTION_ANS)
    original_questions_domains = pd.DataFrame(context_manager.QUESTION_DOMAIN)
    
    updated_questions_ans = preprocessor.all(context_manager.QUESTION_ANS, words.PUNCTUATION)
    updated_questions_domains = preprocessor.all(context_manager.QUESTION_DOMAIN, words.PUNCTUATION)
    # feature_train, feature_test, label_train, label_test = preprocessor.splitter(updated_questions_domains['sentences'], updated_questions_domains['tags'])
    
    # v, feature_train, feature_test = preprocessor.tfidf_vectorizer(feature_train, feature_test)
    # # feature_train = pd.DataFrame(feature_train.todense().tolist(), columns=v.get_feature_names())
    # # feature_test = pd.DataFrame(feature_test.todense().tolist(), columns=v.get_feature_names())
    
    # # from sklearn.linear_model import LogisticRegression
    # # clf = LogisticRegression(C=1.0)
    # # clf.fit(feature_train, label_train)
    # # predictions = clf.predict_proba(feature_test)
    # from sklearn.naive_bayes import MultinomialNB
    # from sklearn import model_selection, naive_bayes, svm	
    # from sklearn.preprocessing import LabelEncoder

    # # clf = MultinomialNB()
    # # clf.fit(feature_train, label_train)
    # # predictions = clf.predict_proba(feature_test)
    # # score = clf.score(feature_test, label_test)
    # # print('Test Accuracy Score ', score)
    # vect = TfidfVectorizer()
    # train = vect.fit_transform(updated_questions_domains['sentences'])	
    # Encoder = LabelEncoder()
    # Train_Y = Encoder.fit_transform(updated_questions_domains['tags'])
    # SVM = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='scale')
    # SVM.fit(train,Train_Y)
    # # Naive = naive_bayes.MultinomialNB()
    # # Naive.fit(feature_train,Train_Y)
    # # predictions_NB = Naive.predict(v.transform(['আপনি এখান থেকে কত দূরে?']))
    # # tag = Encoder.inverse_transform(predictions_NB)
    
    # pro = ' '.join(list(preprocessor.lemmatize('লাভ ইউ'.split(' '))))
    # predictions_SVM = SVM.predict(vect.transform(['পুঁজিবাজার কি']))
    # tag = Encoder.inverse_transform(predictions_SVM)
    # print(list(preprocessor.clean('আপনি এখান থেকে কত দূরে?', punctuation=words.PUNCTUATION)))
    
    
    # from sklearn.metrics import accuracy_score
    # print(accuracy_score(label_test, predictions))
    
    # original_questions_ans.to_excel('original_questions_ans.xlsx', encoding='utf-8')
    # original_questions_domains.to_excel('original_questions_domains.xlsx', encoding='utf-8')
    # updated_questions_ans.to_excel('updated_questions_ans.xlsx', encoding='utf-8')
    # updated_questions_domains.to_excel('updated_questions_domains.xlsx', encoding='utf-8')
    
    # pd.DataFrame(words.get_words()).to_excel('data.xlsx', encoding='utf-8')
    
    # print(f'previous len of {ColorList.bold("questions_ans")} dataset: {ColorList.cyan(len(context_manager.QUESTION_ANS))}')
    # print(f'after doing {ColorList.bold("clean")} and {ColorList.bold("lemmatization")} len of {ColorList.bold("questions_ans")} dataset: {ColorList.cyan(len(questions_ans[0]))}(questions) and {ColorList.cyan(len(questions_ans[1]))}(answers)')
    # print()
    # print('doing train_test_split...\n')
    # questions_train, questions_test, answers_train, answers_test = preprocessor.splitter(questions_ans[0], questions_ans[1])
    # print(f'len of {ColorList.bold("questions_train")}: {ColorList.cyan(len(questions_train))}')
    # print(f'len of {ColorList.bold("questions_test")}: {ColorList.cyan(len(questions_test))}')
    # print(f'len of {ColorList.bold("answers_train")}: {ColorList.cyan(len(answers_train))}')
    # print(f'len of {ColorList.bold("answers_test")}: {ColorList.cyan(len(answers_test))}')


    # v, t, test = preprocessor.tfidf_vectorizer([documentA, documentB], list(documentB))
    # print(pd.DataFrame(t.todense().tolist(), columns=v.get_feature_names()))
    
    test = [['আপনি কখন ক্যাম্পাস ছেড়েছেন?', 'জিপিএসটাইম', 'বাসচ্যাট'], 
            ['আপনার এখানে আসতে কতক্ষণ সময় লাগবে?', 'জিপিএসটাইম', 'বাসচ্যাট'], 
            ['সাধারণত আপনি ক্যাম্পাস থেকে কখন বের হন?', 'সাধারণত সকাল ৮ টার দিকে বের হই।', 'বাসচ্যাট'], 
            ['কখন বের হন?', 'সাধারণত সকাল ৮ টার দিকে বের হই।', 'বাসচ্যাট'], 
            ['আপনি এখন কোথায়?', 'জিপিএসলোকেশন', 'বাসচ্যাট'], 
            ['এই মুহূর্তে আপনার অবস্থান কোথায়?', 'জিপিএসলোকেশন', 'বাসচ্যাট'], 
            ['আপনি এখান থেকে কত দূরে?', 'জিপিএসডিস্টেন্স', 'বাসচ্যাট'], 
            ['আপনার বর্তমান অবস্থান থেকে এখানে আসতে আপনাকে কতক্ষণ সময় নিতে হবে?', 'জিপিএসটাইম', 'বাসচ্যাট'] ]
    # test_df = pd.DataFrame(test, columns=['question', 'ans', 'tag'])	
    # train_data = test_df['question']
    # vect = TfidfVectorizer()
    # train = vect.fit_transform(train_data)	
    # from sklearn import model_selection, naive_bayes, svm	
    # from sklearn.preprocessing import LabelEncoder
    # Encoder = LabelEncoder()
    # Train_Y = Encoder.fit_transform(test_df['ans'])
    # Naive = naive_bayes.MultinomialNB()
    # Naive.fit(train,Train_Y)
    # predictions_NB = Naive.predict(vect.transform(['আপনি এখান থেকে কত দূরে?']))
    # # tag = Encoder.inverse_transform(predictions_NB)
    # SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    # SVM.fit(train,Train_Y)
    # # predict the labels on validation dataset
    # predictions_SVM = SVM.predict(vect.transform(['কখন']))
    # tag = Encoder.inverse_transform(predictions_SVM)
    # print(train.todense())																	

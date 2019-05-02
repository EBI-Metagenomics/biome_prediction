import hashlib
import os
import gzip
import shutil
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import logging

TEST_SIZE = 0.25

PAR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(PAR_DIR, 'data')


class BiomeClassifier:
    input_vectoriser = None
    classifier = None

    def __init__(self, training_data_tsv, verbose):
        self.training_data_tsv = training_data_tsv
        self.training_data_md5 = training_data_tsv + '.md5'
        self.verbose = verbose
        self.classifier = self.gen_model()

    def gen_model(self):
        logging.info('Generating new model....')
        df = self.load_data()
        x_train, x_test, y_train, y_test = self.pre_process_data(df)
        model = self.train(x_train, x_test, y_train, y_test)
        self.write_model_data_md5()
        return model

    @staticmethod
    def train(x_train, x_test, y_train, y_test):
        logging.info('Training (this will take a while)....')
        cls = LogisticRegression(n_jobs=4, solver='lbfgs', multi_class='auto')
        cls.fit(x_train, y_train)
        score = cls.score(x_test, y_test)
        logging.info('Training complete!')
        logging.info('Accuracy:', score)
        return cls

    def interactive(self):
        while True:
            inp = input('Please enter text to classify: ')
            pred_probs = self.pred_input(inp)
            print('Prediction: {}'.format(pred_probs[0]))
            for cls, pred2 in pred_probs:
                print('{}: {}'.format(cls, pred2))

    def pred_input(self, inp):
        df = pd.DataFrame()
        df['input'] = [inp]
        i = self.input_vectoriser.transform(df['input'])
        pred = self.classifier.predict_proba(i)
        class_preds = zip(self.classifier.classes_, pred[0])
        class_preds = sorted(class_preds, key=lambda r: r[1], reverse=True)[0:5]
        return class_preds

    def load_data(self):
        logging.info('Loading training data...')
        self.get_training_data_file()

        df = pd.read_csv(self.training_data_tsv, sep='\t')
        df = df.dropna()
        return df

    def pre_process_data(self, df):
        logging.info('Pre-processing training data...')
        df['input'] = df['STUDY_NAME'] + df['STUDY_ABSTRACT'] + df['SAMPLE_DESC'] + df['SAMPLE_NAME'] + df[
            'SAMPLE_ALIAS']
        del df['STUDY_NAME']
        del df['STUDY_ABSTRACT']
        del df['SAMPLE_DESC']
        del df['SAMPLE_NAME']
        del df['SAMPLE_ALIAS']
        df = df[df.groupby('LINEAGE').LINEAGE.transform(len) > 1]
        x_train, x_test, y_train, y_test = train_test_split(df['input'], df['LINEAGE'], test_size=TEST_SIZE,
                                                            random_state=1000, stratify=df['LINEAGE'])
        self.input_vectoriser = CountVectorizer(min_df=0, lowercase=True)
        self.input_vectoriser.fit(x_train)
        x_train = self.input_vectoriser.transform(x_train)
        x_test = self.input_vectoriser.transform(x_test)
        return x_train, x_test, y_train, y_test

    def write_model_data_md5(self):
        md5 = calc_md5(self.training_data_tsv)
        with open(self.training_data_md5, 'w') as f:
            f.write(md5)

    def check_model_data_md5(self):
        self.get_training_data_file()
        try:
            md5 = calc_md5(self.training_data_tsv)
            with open(self.training_data_md5) as f:
                curr_md5 = f.read().strip()
            return md5 == curr_md5
        except FileNotFoundError:
            return False

    def get_training_data_file(self):
        if not os.path.exists(self.training_data_tsv):
            logging.warning('Could not find raw data file, attempting to decompress archive...')
            decompress(self.training_data_tsv + '.gz', self.training_data_tsv)


def calc_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def decompress(fname, dest):
    with gzip.open(fname, 'rb') as f_in:
        with open(dest, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


if __name__ == '__main__':
    main()

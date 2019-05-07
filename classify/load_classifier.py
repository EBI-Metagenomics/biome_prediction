from joblib import dump, load
import logging
import argparse
import os
from subprocess import check_output, CalledProcessError

from classify.classifier import BiomeClassifier

PAR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PAR_DIR, 'data')

FETCH_MODEL_SCRIPT = os.path.join(DATA_DIR, 'fetch_ftp_models.sh')
TRAINING_DATA = os.path.join(DATA_DIR, 'raw_data.tsv')

MD5_FILE = TRAINING_DATA + '.md5'

MODEL_PICKLE = os.path.join(DATA_DIR, 'model.p.gz')

INPUT_TRANSFORMER_PICKLE = os.path.join(DATA_DIR, 'input_trans.p.gz')


def save(self):
    logging.info('Saving new model....')
    self.__module__ = 'classify'
    dump(self, MODEL_PICKLE, compress=9)


def load_model():
    logging.info('Attempting to load model from pickle file...')
    logging.info(MODEL_PICKLE)
    return load(MODEL_PICKLE)


def load_or_fetch_ftp_model():
    bc = None
    try:
        bc = load_model()
    except (FileNotFoundError, AttributeError, ValueError) as e:
        print(e)
        try:
            bc = fetch_ftp_models()
            if not isinstance(bc, BiomeClassifier):
                print(bc)
                raise EnvironmentError('Invalid pickle file')
        except (EnvironmentError, CalledProcessError, AttributeError) as e:
            logging.error(e)
            bc = None
    return bc


def fetch_ftp_models():
    logging.info('Attempting to fetch model from ftp...')
    check_output(['bash', FETCH_MODEL_SCRIPT])
    return load_model()


def parse_args():
    parser = argparse.ArgumentParser(description='Biome classification tool using the GOLD ontology')
    parser.add_argument('-r', '--regenerate-model', action='store_true', help='Regenerate model from data')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()


def get_model(regenerate=False, verbose=False):
    if not regenerate:
        bc = load_or_fetch_ftp_model()

    # Fallback if loading failed
    if not bc or regenerate:
        bc = BiomeClassifier(TRAINING_DATA, verbose)
    return bc


def main():
    args = parse_args()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG if args.verbose else logging.INFO)
    bc = get_model(args.regenerate_model, args.verbose)
    bc.interactive()


if __name__ == '__main__':
    main()

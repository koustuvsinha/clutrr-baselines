# Check the data file for duplicate entities within the same sentence
import pandas as pd
import re
from nltk.tokenize import word_tokenize, sent_tokenize
import argparse
from tqdm import tqdm

def check_sent(text):
    sents = sent_tokenize(text)
    for sent_idx, sent in enumerate(sents):
        ents = re.findall('\[(.*?)\]', sent)
        if len(ents) != len(set(ents)):
            print("We have a corrupt record")
            print("In sentence number : {}".format(sent_idx))
            print("Offending sentence: {}".format(sent))
            return False
    return True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="", type=str, help='Filename to check')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print("Reading file {}".format(args.file))
    df = pd.read_csv(args.file, comment='#')
    print("Starting to check ...")
    pb = tqdm(total=len(df))
    for i,row in df.iterrows():
        if not check_sent(row['story']):
            raise AssertionError("File corrupt at story for line {}".format(i))
        if not check_sent(row['summary']):
            raise AssertionError("File corrupt at summary for line {}".format(i))
        pb.update(1)
    pb.close()
    print("Check done")

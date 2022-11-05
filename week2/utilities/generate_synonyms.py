import argparse
import os
import fasttext


parser = argparse.ArgumentParser(description='Generates Synonyms for given list of words in a file.')
general = parser.add_argument_group("general")
general.add_argument("--model", default="/workspace/datasets/fasttext/title_model.bin", \
    help="The trained model file name")
general.add_argument("--threshold", default=0.75, type=float, \
    help="The threshold similarity to filter similarity words")
general.add_argument("--input", default="/workspace/datasets/fasttext/top_words.txt",  \
    help="The input file to read")
general.add_argument("--output", default="/workspace/datasets/fasttext/synonyms.csv", \
    help="The output csv file to write the filtered labels")

args = parser.parse_args()

model = fasttext.load_model(args.model)

with open(args.input, 'r') as inputFile:
    with open(args.output, 'w') as outputFile:
        for word in inputFile:
            similarityWords = model.get_nearest_neighbors(word)
            count = 0
            synomyms = {}
            synomyms[word.rstrip()] = 1
            for similarityWord in similarityWords:
                if similarityWord[0] >= args.threshold and similarityWord[1].rstrip() not in synomyms:
                    synomyms[similarityWord[1].rstrip()] = 1
            if len(synomyms) > 1:
                outputFile.write(",".join(synomyms.keys()) + '\n')
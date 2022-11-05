import argparse
import os

parser = argparse.ArgumentParser(description='Prune labels by number of products.')
general = parser.add_argument_group("general")
general.add_argument("--threshold", default=500, type=int, \
    help="The minimum number of products associated with a lable to keep")
general.add_argument("--input", default="/workspace/datasets/fasttext/labeled_products.txt", \
    help="The input file to read")
general.add_argument("--output", default="/workspace/datasets/fasttext/pruned_labeled_products.txt", \
    help="The output file to write the pruned labeled products")

args = parser.parse_args()

# Count the number of products by label
productCounts = {}
with open(args.input, 'r') as inputFile:
    for line in inputFile:
        fields = line.split()
        if fields[0] not in productCounts:
            productCounts[fields[0]] = 1
        else:
            productCounts[fields[0]] += 1

prunedLabels = {}
for productCount in productCounts.items():
    if productCount[1] >= args.threshold:
        prunedLabels[productCount[0]] = productCount[1]

with open(args.input, 'r') as inputFile:
    with open(args.output, 'w') as outputFile:
        for line in inputFile:
            fields = line.split()
            if fields[0] in prunedLabels:
                outputFile.write(line)

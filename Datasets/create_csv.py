import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import json
from empath import Empath
from util import contentToString, getSegments
import csv
from util import isGood
from pathlib import Path
from tqdm import tqdm
from extract_features import extractFeatures

parser = argparse.ArgumentParser(description='Create a tsv dataset from a datapack')
parser.add_argument(
	"--dataset",
	dest='dataset',
	help="from which dataset the datapack is",
	required=True
)
parser.add_argument(
	"--datapackID",
	dest='datapackID',
	help="the unique identifier that the datapack is going to have"
)
parser.add_argument('--all', action=argparse.BooleanOptionalAction)
args = parser.parse_args()
if args.datapackID is None: args.datapackID = args.dataset

print("\n---           Arguments           ---")
for key, val in args.__dict__.items(): print("%20s: %s" % (key, val))
print()

def writeCSV(file, datapack, datasetType, extra_features, categories):
	writer = csv.writer(file, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
	writer.writerow(["label", "chatName", "segment"] + extra_features) # header

	for chatName, chat in tqdm(datapack["chats"].items()):
		for i, segment in enumerate(getSegments(chat)):
			if not isGood(segment, args.dataset): continue

			# add segment number if segment is not whole chat
			segmentName = "%s-%s" % (chatName, i) \
				if len(segment) != len(chat["content"]) else chatName

			writer.writerow([chat["className"], segmentName, contentToString(segment)] + extractFeatures(segment, categories))
		#return ######################


lexicon = Empath()
res = lexicon.analyze('this is a sentence')
categories = list(res.keys())

# Top 15 (they are sorted)
if not args.all:
	categories = categories[:15]

extra_features = ['length_rel', 'length_abs', 'num_questions_rel', 'num_questions_abs', 'size_of_words_rel', 'size_of_words_abs']
extra_features += ['empath_' + x for x in categories]

for datasetType in ["train", "test"]:
	datapackPath = "%s/datapacks/datapack-%s-%s.json" % (
		args.dataset, args.datapackID, datasetType)

	outPath = os.path.join("%s/csv/" % args.dataset)
	Path(outPath).mkdir(parents=True, exist_ok=True)
	csvPath = os.path.join(outPath, "%s-%s.csv" %
		(args.datapackID, datasetType))
	with open(datapackPath, "r") as file:
		datapack = json.load(file)

		# write TSV
		with open(csvPath, 'w', newline='') as file:
			writeCSV(file, datapack, datasetType, extra_features, categories)

print("\nSuccessfully wrote all csv files")
import json

MESSAGE_DELIMITER = " "

def contentToString(content):
	string = ""
	for ct in content:
		if not isNonemptyMsg(ct) or ct["type"] != "message": continue
		if string != "": string += MESSAGE_DELIMITER
		string += ct["body"]
	return string

def contentToJson(content):
	messages = []
	for ct in content:
		if not isNonemptyMsg(ct) or ct["type"] != "message": continue
		#if messages != "": messages += MESSAGE_DELIMITER
		messages += [ct["body"]]
	return json.dumps(messages)

def getSegments(chat):
	currentSegmentStart = 0
	for i, ct in enumerate(chat["content"] + [{"type": "separator"}]):
		if ct is None or ct["type"] != "separator": continue # await separator

		segment = chat["content"][currentSegmentStart:i]
		yield segment
		currentSegmentStart = i+1

PANC_MIN_MSG_NUM = 6
PANC_MAX_MSG_NUM = 150

# TODO refactor
def isGood(segment, dataset): # which segments should be filtered
	if dataset == "PANC":
		# The negative PAN12 segments have already been filtered in
		# PANC/create_datapack.py. This code is only for the segments from
		# the complete positive chats originally from CC2.
		numOfNonemptyMessages = sum([isNonemptyMsg(msg) for msg in segment])
		return PANC_MIN_MSG_NUM <= numOfNonemptyMessages <= PANC_MAX_MSG_NUM
	if dataset == "VTPAN": return True # don't filter these
	return True # for other datasets also allow all segments

def isNonemptyMsg(ct):
	return ct is not None and ct["type"] == "message" and bool(ct["body"])

from datetime import datetime, timezone

def getUNIXTimestamp():
	return int(datetime.now().replace(tzinfo=timezone.utc).timestamp()*1000)

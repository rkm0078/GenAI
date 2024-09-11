import random
import json
import pickle
import numpy as np
import tensorflow
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intent = json.load(open(r'C:\Projects\GenAI\Gen_AI_NULLCLASS\NLP_ChatBot\Data\intents.json')).read()

words = []
classes = []
documents = []
ignoreLetters = ["?","!",".",","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append(wordList, intent["tags"])

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append.append(0)

outputRow = list(outputEmpty)
outputRow[classes.index(document[1])] = 1

training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:,: len(word)]
trainY = training[: len(word)]
        
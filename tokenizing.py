import nltk
nltk.download('wordnet')
nltk.download('punkt')

o_henry_story = open("the_gift_of_the_magi.txt", "r").read()
sentences = nltk.sent_tokenize(o_henry_story)


# tokenizing examples
random_sentence = sentences[101]
print(random_sentence)

whitespace_tokenizer = nltk.tokenize.WhitespaceTokenizer()
print(whitespace_tokenizer.tokenize(random_sentence))

punct_tokenizer = nltk.tokenize.WordPunctTokenizer()
print(punct_tokenizer.tokenize(random_sentence))

treebank_tokenizer = nltk.tokenize.TreebankWordTokenizer()
print(treebank_tokenizer.tokenize(random_sentence))


# stemming examples
random_sentence_2 = sentences[19]
print(random_sentence_2)
tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokens = tokenizer.tokenize(random_sentence_2)

stemmer = nltk.stem.PorterStemmer()
porter_stemmed = " ".join(stemmer.stem(token) for token in tokens)
print(porter_stemmed)

stemmer = nltk.stem.WordNetLemmatizer()
wordnet_lemmatized = " ".join(stemmer.lemmatize(token) for token in tokens)
print(wordnet_lemmatized)
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

text = 'This is a test. Of a good review maybe.sorta'

sentoke = sent_tokenize(text)
print sentoke


wordoke = word_tokenize(sentoke[0])
print wordoke


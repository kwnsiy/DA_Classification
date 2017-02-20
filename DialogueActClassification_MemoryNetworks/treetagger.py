# coding:utf-8

import treetaggerwrapper
tagger = treetaggerwrapper.TreeTagger(TAGLANG="en")

def parse(sentence):
  sentence =  unicode(sentence.encode("utf-8", errors="ignore"), errors="ignore")
  res = tagger.TagText(unicode(sentence))
  return [x for x in res if len(x.split("\t")) == 3]
  
# coding:utf-8

doc=
"""
  Response Generation Module Clustering for Infomation-Seeking Dialogue System
  2017/02/15 ~
  ・同一話者による連続発話を1発話に統合
  ・1ファイルに必要な情報をすべてマージする
  ・各発話に，IDを振り素性を参照できるようにする
"""

include("switch_depwarn.jl")
switch_depwarn!(false)

using DataFrames
using DataStructures
using PyCall
using ScikitLearn
using JSON
using ScikitLearn: fit!, predict, transform

@sk_import cluster: KMeans
@sk_import metrics: classification_report

# コーパス読み込み
d = open(deserialize, "./data/corpus.dic")

# 対話ID
session_ids = collect(keys(d))

# unigram & bigram 辞書
function word_dict(d, session_ids)
  word_d = Dict(:unigram => [], :bigram => [], :pos => [])
  for ids in session_ids
     for i in  collect(length(d[ids][:unigram]))
       for x in d[ids][:unigram][i]; push!(word_d[:unigram], x) ; end
       for x in d[ids][:bigram][i]; push!(word_d[:bigram], x) ; end
       for x in d[ids][:pos][i]; push!(word_d[:pos], x) ; end
    end
  end
  return Dict(:unigram => counter(word_d[:unigram]), 
    :bigram => counter(word_d[:bigram]), :pos => counter(word_d[:pos]))  
end

word_d = word_dict(d, session_ids)
d[:unigram] = word_d[:unigram]
d[:bigram] = word_d[:bigram]


# 対話行為ラベル
df = readtable("./data/dialogueactlabel.dat", separator = '\t', header = true)
da_dic = Dict()
da_dic[2000] = ["NULL","NULL"]
for i in collect(1:length(df[1]))
  da = df[:dialogueActLabelId][i]
  label = df[:label][i]
  label_nosub = df[:label][i]
  if length(split(label_nosub, ':')) == 3
    label_nosub = join(split(label_nosub, ':')[1:2],':')
  end
  da_dic[da] = [label_nosub, label]
end

# 追加情報
for sid in session_ids
  normarized_sequence, message_length = [], []
  da_function_label, da_domain_label = [], []
  for (i,cont) in enumerate(d[sid][:unigram])
    push!(normarized_sequence, d[sid][:sequence][i]/length(cont))      
    push!(message_length, length(cont))
    push!(da_function_label, da_dic[d[sid][:function_da][i]])
    push!(da_domain_label, da_dic[d[sid][:domain_da][i]])
    d[sid][:speaker][i] = d[sid][:speaker][i] - 1
  end
  d[sid][:norm_sequence] = normarized_sequence
  d[sid][:message_length] = message_length
  d[sid][:function_da_label] = da_function_label
  d[sid][:domain_da_label] = da_domain_label
end

# corpus dictionary
open(io -> serialize(io, d), "./data/annotated_corpus.dic", "w")
JSON.print(open("./data/annotated_corpus.json", "w"), JSON.json(d))


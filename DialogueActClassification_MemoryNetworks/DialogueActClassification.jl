# coding:utf-8

doc=
"""
  Dialogue Act Classification Based on Memory Networks
  修正
  2017/02/05 ~
"""

include("switch_depwarn.jl")
switch_depwarn!(false)

using DataFrames
using DataStructures
using PyCall
using ScikitLearn

# python module
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport prep
@pyimport memnet

# sklearn module
@sk_import metrics: classification_report

# 単語->単語インデックス
function convert_sequence(word_sequence, word_d, n_count)
  rank =  Dict(w[1] => i for (i, w) in enumerate(sort(collect(word_d), by=x->x[2], rev = true)))
  for (i, seq) in enumerate(word_sequence)
      for (j,w) in enumerate(seq)
        if word_d[w] > n_count
          word_sequence[i][j] = rank[w]
        else
          word_sequence[i][j] = 0
        end
      end
    end
  return word_sequence
end

# 学習用データ作成
function make_feature(d, session_ids)
  stories, questions = [], []
  da_function, da_domain = [], []
  @printf "push ...\n"
  for sid in session_ids
    dialogue_history = []
    for i in collect(1:length(d[sid][:utterance]))
      if i == 1
        dialogue_history = vcat(dialogue_history, ["<S>", "</S>"])
      else
        dialogue_history = vcat(dialogue_history, d[sid][:unigram][i-1])
      end
      push!(stories, dialogue_history)
      push!(questions, d[sid][:unigram][i])
      push!(da_function, d[sid][:function_da_label][i][1])
      push!(da_domain, d[sid][:domain_da_label][i][1])      
    end
  end
  @printf "to sequence ...\n"
  stories = convert_sequence(stories, d[:unigram], 0)    
  questions = convert_sequence(questions, d[:unigram], 0)   
  @printf "pad sequence ...\n"
  story_maxlen = 200 #maximum([length(x) for x in stories])
  query_maxlen = 200 #maximum([length(x) for x in questions])
  stories = prep.pad_sequence(stories, story_maxlen)
  questions = prep.pad_sequence(questions, query_maxlen)
  return Dict(:stories => stories, :questions =>  questions, :da_domain => da_domain, :da_function => da_function)
end

function main(tgt)
  @printf "read corpus ...\n"
  d = open(deserialize, "./data/annotated_corpus.dic")
  session_ids = filter(x -> !(in(x, [:unigram, :bigram])), collect(keys(d)))
  @printf "make feature ...\n"
  feature =  make_feature(d, session_ids)
  @printf "dummy variable...\n"
  feature[tgt], ans_label = prep.get_dummies(feature[tgt])
  @printf "10 fold cross validation ...\n"
  data = prep.cross_validation(collect(1:length(feature[tgt][:, 1])), 10)
  for (i,(train_id, test_id)) in enumerate(zip(data[:,1], data[:, 2]))
    println("---------- $i ----------")
    @printf "split feature ...\n"
    train_feature, test_feature = Dict(), Dict()
    for k in keys(feature)
      train_feature[k]  = feature[k][train_id, :]
      test_feature[k] =  feature[k][test_id, :]
    end
    @printf "read model ...\n"
    vocab_size = length(collect(keys(d[:unigram])))+1
    story_maxlen, query_maxlen = length(feature[:stories][1, :]), length(feature[:questions][1,:])
    answer_dim, embedding_dim = length(feature[tgt][1, :]), 128
    batch_size, nb_epoch = 64, 15 # 256 30 64 15
    @show vocab_size; @show story_maxlen; @show query_maxlen; @show answer_dim
    @show batch_size; @show nb_epoch
    model = memnet.memnet(vocab_size, story_maxlen, query_maxlen, answer_dim, embedding_dim)
    @printf "model fit ...\n"
    model = memnet.best_fit(model, batch_size, nb_epoch, train_feature, test_feature, tgt)
    @printf "predict ...\n"
    memnet.predict(model, test_feature, tgt)
  end
end

main(:da_domain)

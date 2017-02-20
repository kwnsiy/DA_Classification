# coding:utf-8

doc=
"""
  Dialogue Act Classification for Infomation-Seeking Dialogue System
  修正予定
  2017/02/05 ~
"""

include("switch_depwarn.jl")
switch_depwarn!(false)

using DataFrames
using DataStructures
using JSON
using PyCall
using MLBase
using ScikitLearn

using ScikitLearn: fit!, predict, transform
@sk_import ensemble: RandomForestClassifier
@sk_import ensemble: GradientBoostingClassifier
@sk_import ensemble: RandomTreesEmbedding
@sk_import metrics: classification_report
@sk_import neighbors: KNeighborsClassifier
@sk_import neural_network: MLPClassifier
@sk_import gaussian_process: GaussianProcessClassifier
@sk_import svm: SVC

# python module
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport prep
@pyimport numpy as np

# 追加素性作成
# normarized sequence number & message length
function add_feature(d, session_ids)
  for sid in session_ids
    normarized_sequence, message_length = [], []
    for (i,cont) in enumerate(d[sid][:unigram])
      push!(normarized_sequence, d[sid][:sequence][i]/length(cont))      
      push!(message_length, length(cont))
    end
    d[sid][:norm_sequence] = normarized_sequence
    d[sid][:message_length] = message_length
  end
  return d
end

# 出現回数でフィルタリング
function convert_sequence(word_sequence, word_d, n_count)
  # rank_dict
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

# 素性作成
function make_feature(d, word_d, session_ids, bowlen, boblen, poplen)
  unigram, bigram, sequence, norm_sequence, message_length, pos, speaker = [], [], [], [], [], [], []
  da_function, da_domain = [], []
  da_function_label, da_domain_label = [], []
  @printf "push ...\n"
  for sid in session_ids
    for i in collect(1:length(d[sid][:utterance]))
      push!(unigram, d[sid][:unigram][i]), push!(bigram, d[sid][:bigram][i]), push!(sequence, d[sid][:sequence][i])
      push!(norm_sequence, d[sid][:norm_sequence][i]), push!(message_length, d[sid][:message_length][i])
      push!(da_function, d[sid][:function_da][i]), push!(da_domain, d[sid][:domain_da][i]), push!(pos, d[sid][:pos][i])
      push!(speaker, d[sid][:speaker][i] - 1)
      push!(da_function_label, d[sid][:function_da_label][i][1]), push!(da_domain_label, d[sid][:domain_da_label][i][1])
    end
  end
  @printf "to sequence ...\n"
  unigram = convert_sequence(unigram, word_d[:unigram], 0)
  bigram = convert_sequence(bigram, word_d[:bigram], 0)
  pos = convert_sequence(pos, word_d[:pos], 0)
  @printf "to bag-of-words ...\n"
  bow  = prep.sequences_to_matrix(unigram, bowlen, "count")
  @printf "to bag-of-bigram ...\n"
  bob = prep.sequences_to_matrix(bigram, boblen, "count")
  @printf "to bag-of-pos ...\n"
  bop = prep.sequences_to_matrix(pos, poplen, "count")    
  return Dict(:unigram => unigram, :bigram =>  bigram, :sequence => sequence, :norm_sequence => norm_sequence, :da_domain_label => da_domain_label, :da_function_label => da_function_label,
   :message_length => message_length, :bow => bow, :bob => bob, :da_domain => da_domain, :da_function => da_function, :pos => pos, :bop => bop, :speaker => speaker)
end

# 単語出現頻度
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

# 論文記載以外のDAが定義された発話をスキップ
function skip_label(feature)
  skipindex = []
  skipindex! = []  
  for i in collect(1:length(feature[:da_function]))
    in(feature[:da_function][i], [201, 192]) || feature[:da_domain_label][i] == "NULL"  && push!(skipindex, i)
    !(in(feature[:da_function][i], [201, 192]) || feature[:da_domain_label][i] == "NULL") && push!(skipindex!, i)
  end
  for k in keys(feature)
    feature[k] = feature[k][skipindex!, :]
  end
  return feature, skipindex!
end

# 素性作成
function get_features()
  @printf "read corpus ...\n"
  d = open(deserialize, "./data/annotated_corpus.dic")
  @printf "add feature ...\n"
  session_ids = filter(x -> !(in(x, [:unigram, :bigram])), collect(keys(d)))
  d = add_feature(d, session_ids)
  @printf "make feature ...\n"
  word_d = word_dict(d, session_ids)
  feature = make_feature(d, word_d, session_ids, int(length(keys(word_d[:unigram]))), int(length(keys(word_d[:bigram]))), int(length(keys(word_d[:pos]))))
  return d, word_d, feature
end

# baseline
function baseline_randomforest(tgt)
  @printf "preprocessing ..\n"
  corpus_d, word_d, feature = get_features()
  @printf "skip label ...\n"
  feature, skipindex! = skip_label(feature)
  @printf "make kfold validation dataset ...\n"  
  data = prep.cross_validation(collect(1:length(feature[tgt])), 10)
  @printf "start 10 fold validation ...\n"
  for line_id in zip(data[:,1], data[:, 2])
    train_id, test_id = line_id[1], line_id[2]
    @printf "split feature ...\n"
    train_feature, test_feature = Dict(), Dict()
    for k in keys(feature)
      train_feature[k]  = feature[k][train_id, :]
      test_feature[k] =  feature[k][test_id, :]
    end
    @printf "train model ...\n"
    train = hcat(train_feature[:bow], train_feature[:bob], train_feature[:speaker], train_feature[:norm_sequence], train_feature[:message_length])
    test = hcat(test_feature[:bow], test_feature[:bob], test_feature[:speaker], test_feature[:norm_sequence], test_feature[:message_length])
    model = RandomForestClassifier(n_estimators=200, max_depth=100, max_features="auto", criterion="gini") # n_estimator = 200 max_deps=200 auto
    model = fit!(model, train, vec(train_feature[tgt]))
    @printf "predict ...\n"
    output = predict(model, test)
    println(classification_report(vec(test_feature[tgt]), output, digits=4))
    #return
  end
end

# proposal
function proposal_neuralnet(tgt)
  maxlen = 200
  @printf "preprocessing ..\n"
  corpus_d, word_d, feature = get_features()
  @printf "skip label ...\n"
  feature, skipindex! = skip_label(feature)
  @printf "pad sequence ...\n"
  feature[:unigram] = prep.pad_sequence(feature[:unigram], maxlen)
  feature[:bigram] = prep.pad_sequence(feature[:bigram], maxlen)
  feature[:pos] = prep.pad_sequence(feature[:pos], maxlen)
  @printf "dummy variable...\n"
  feature[tgt] = vec(feature[tgt])
  feature[tgt], d = prep.get_dummies(feature[tgt])
  @printf "make kfold validation dataset ...\n"  
  data = prep.cross_validation(collect(1:length(feature[tgt][:, 1])), 10)
  @printf "start 10 fold validation ...\n"
  for line_id in zip(data[:,1], data[:, 2])
    train_id, test_id = line_id[1], line_id[2]
    @printf "split feature ...\n"
    train_feature, test_feature = Dict(), Dict()
    for k in keys(feature)
      train_feature[k]  = feature[k][train_id, :]
      test_feature[k] =  feature[k][test_id, :]
    end
    @printf "read model ...\n"
    # dense_dim, embedding_dim, sequence_maxlen, max_features_uni, max_features_pos, max_features_bi, bowlen, boblen, boplen, glove_embeddings
    bowlen, boblen, boplen = length(feature[:bow][1, :]), length(feature[:bob][1, :]), length(feature[:bop][1, :])
    glove_embeddings = ""
    #glove_embeddings = Dict(i => lowercase(w[1]) for (i, w) in enumerate(sort(collect(word_d[:unigram]), by=x->x[2], rev = true)))
    model = prep.nn_classifier(length(feature[tgt][1, :]), 200, maxlen, bowlen, boblen, boplen, bowlen, boblen, boplen, glove_embeddings)
    @printf "model fit ...\n"
    batch_size, nb_epoch = 128, 30 # 216 25 dim: 200
    feature_list =  ["bow", "bob", "speaker", "norm_sequence", "message_length"]
    model = prep.best_fit(model, 1, train_feature, test_feature, string(tgt), feature_list, batch_size, nb_epoch)
    @printf "predict ...\n"
    prep.predict(model, test_feature, tgt, feature_list)
    #println(classification_report(vec(test_feature[:da_function]), output, digits=4))
  end
end

# baseline
# baseline_randomforest(:da_domain)

# proposal
proposal_neuralnet(:da_domain_label)

# baseline
# acc .8574
# acc 

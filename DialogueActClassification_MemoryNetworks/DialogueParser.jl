# coding:utf-8

include("switch_depwarn.jl")
switch_depwarn!(false)

using DataFrames
using DataStructures
using PyCall
using JSON

unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport treetagger

doc=
"""
アノテーション済みセッションの抽出
json形式出力
前処理：
  urlをreplaced-url
  数字を@card@
形態素解析：
  見出し語化　
  品詞タグセットはtree-taggerタグセット
"""

#parser = "nltk"
parser = "tagger"

# dataframes
da = readtable("./data/dialogueact.dat", separator = '\t')
dialogue = readtable("./data/message.dat", separator = '\t')

# tokenizer
@pyimport nltk

# ngram
function ngram(string, n)
  words = []
  !contains(summary(string),"Array") && (string = utf32(string))
  for i in collect(1:(length(string)-n+1))
    push!(words, string[i:i+n-1])
  end
  return words
end

# unigram & bigram
@show "creating ngram feature ..."
unigram, bigram, pos = [], [], []
for (i, message) in enumerate(dialogue[:content])
  i % 1000 == 0 && println(i)
  message =join(map(x->replace(x,r"^http.+","replaced-url"), vcat(split(message))), " ")
  token = []
  if parser == "tagger"
    seg = [(x[3], x[2]) for x in map(w->split(w, "\t"), treetagger.parse(message))]
    pos_tag = [x[2] for x  in seg]
    token = vcat("<s>", vcat([x[1] for x  in seg]), "</s>")
    push!(pos, pos_tag)
  else
    token = vcat("<s>", vcat(nltk.word_tokenize(message), "</s>"))
    map!(x -> replace(x, r"^\d+$", "9999"), token)
  end
  push!(unigram, token)
  push!(bigram, map(x -> join(x, "@"), ngram(token, 2)))
end
parser == "tagger" && (dialogue[:pos] = pos)
dialogue[:unigram] = map(x->map(uppercase, x), unigram)
dialogue[:bigram] = map(x->map(uppercase, x), bigram)

@show "extracting annotated session ..."
d, c = Dict(), 0
for sid in collect(1:800)
  # 8発話以上のセッション抽出
  df_message = dialogue[dialogue[:conversationId] .== string(sid), :]
  length(df_message[1]) < 8 && continue
  function_da, domain_da, utterance, speaker, sequence, unigram, bigram, pos, i = [], [], [], [], [], [], [], [], 0
  for (i, mid) in enumerate(df_message[:messageId])
    df_da = da[da[:MessageId] .== mid, :]
    # NAを置き換え
    df_da[isna(df_da[:functionDALabelId]), :functionDALabelId] = 1000    
    df_da[isna(df_da[:domainDALabelId]), :domainDALabelId] = 2000
    length(df_da[1]) == 0 && break
    # push
    push!(function_da, sort(collect(counter(df_da[:functionDALabelId])), by = x->x[2])[end][1])
    push!(domain_da, sort(collect(counter(df_da[:domainDALabelId])), by = x->x[2])[end][1])
    push!(speaker, parse(df_message[:speaker][i]))
    push!(sequence, parse(df_message[:sequence][i]))
    push!(utterance, df_message[:content][i])
    push!(unigram, df_message[:unigram][i])
    push!(bigram, df_message[:bigram][i])
    parser == "tagger" &&  push!(pos, df_message[:pos][i])
  end
  # すべてアノテーション済みの場合
  if length(df_message[1]) == i
    if parser == "tagger"
      d[sid] = Dict(:function_da => [], :domain_da => [], :utterance => [], :speaker => [],
        :sequence => [], :unigram => [], :bigram => [], :pos => [])
    else
      d[sid] = Dict(:function_da => [], :domain_da => [], :utterance => [], :speaker => [],
        :sequence => [], :unigram => [], :bigram => [])
    end
    tmp = collect(1:length(sequence))
    # sort
    for (i, seq) in enumerate(sort(sequence, rev=true))
      tgt = tmp[i]
      push!(d[sid][:function_da], function_da[tgt])
      push!(d[sid][:domain_da], domain_da[tgt])
      push!(d[sid][:utterance], utterance[tgt])
      push!(d[sid][:speaker], speaker[tgt])
      push!(d[sid][:sequence], i)
      push!(d[sid][:unigram], unigram[tgt])
      push!(d[sid][:bigram], bigram[tgt])
      parser == "tagger" &&  push!(d[sid][:pos], pos[tgt])
    end
    c += 1
    c % 10 == 0 && println(c)
  end
end

# シリアライズ
open(io -> serialize(io, d), "./data/corpus.dic", "w")
JSON.print(open("./data/corpus.json", "w"), JSON.json(d))

# デシリアライズ
# d = open(deserialize, "corpus.dic")
# d = JSON.parse(JSON.parsefile("annotated_session.json", dicttype=Dict))

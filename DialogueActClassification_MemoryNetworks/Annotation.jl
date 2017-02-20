# coding:utf-8

include("switch_depwarn.jl")
switch_depwarn!(false)

using DataFrames
using DataStructures
#using PyCall
#@pyimport polyglot.text as text

function input(prompt::AbstractString="")
  print(prompt)
  return chomp(readline())
end

# データ読み込み
d = open(deserialize, "./data/annotated_corpus.dic")
session_ids = filter(x -> !(in(x, [:unigram, :bigram])), collect(keys(d)))

# 記録用
if isfile("./data/simple_annotation.dic")
  log = open(deserialize, "./data/simple_annotation.dic")
else
  log = Dict()
end  

# 一発話+segmentごとにデータを見ていく
for (i,sid) in enumerate(session_ids)
  # 登録済みならパス
  !haskey(log, sid) && (log[sid] = Dict())
  for j in collect(1:length(d[sid][:utterance]))
   ant, blob, spk, spkd = [], d[sid][:utterance][j],d[sid][:speaker][j], Dict(0=>"Librarian",1=>"Patron")
   dm, fc = d[sid][:domain_da_label][j],d[sid][:function_da_label][j]
   println("---------- Session $i: $j ----------")
   println(fc[2], " & ",dm[2])
   println(spkd[spk],": ",blob)   
   while true
      println(ant)
      usrin = input(">>")
      usrin == "exit" && exit()
      if usrin == "" 
        println(ant)
        chk = input("ok?")
        !in(chk, ["ok","y","yes",""]) && continue
        log[sid][i] = ant
        break
      elseif usrin == "back"
        pop!(ant)
      else
        push!(ant, usrin)
      end
    end
  end
  # 保存
  open(io -> serialize(io, log), "./data/simple_annotation.dic", "w")
end


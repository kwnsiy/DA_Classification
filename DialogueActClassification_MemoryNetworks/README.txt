# ParseDialogue.jl
Inoueのデータを読み込み辞書形式に変換
ついでに、形態素解析も実施し、unigram、bigramの情報も追加する

# ConvertDialogue.jl
上記データに対して、追加の素性とその他情報（Unigram辞書, Bigram辞書）を付与

# memnn.py
メモリーネットワーク実装
story question answer が学習に必要
questionに対して応答となるanswerを出力するモデル
questionはメモリに格納されたstoryの情報を考慮することができる
詳細は謎

# このディレクトリでやろうとしていること
「memorynetworkを用いた対話行為推定」
story: 直前までの発話の系列，question: 今の発話， answer: 今の発話の対話行為
とすることで，対話行為推定できないか？
直感的には、できそうな気がするので、とりあえずやってみた


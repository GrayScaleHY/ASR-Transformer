# ASR-Transformer
Transformer的语音识别模型

# Requirements
pytorch >= 1.2.0
Torchaudio >= 0.3.0
py37.yml中包含了train需要的python包

# Train
1. 将需要准备wav音频的索引和标注文件(wav.scp、text、vocab).格式和kaldi的一样
2. 根据自己的需求修改模型config文件，./config/transformer.ymal
3. 运行run.py

# Eval
运行inference.py。
以下是链接中可以下载已训练好的模型。该模型识别效果优于百度云短句识别效果。
https://download.csdn.net/download/qq_41854731/19038018

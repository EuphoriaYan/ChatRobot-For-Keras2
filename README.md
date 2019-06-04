# ChatRobot-Keras2

## 1. 效果展示  
### 1.0 `python train.py`执行效果图  
![image](https://github.com/shen1994/README/raw/master/images/ChatRobot_train.jpg)  
### 1.1 `python test.py`执行效果图  
![image](https://github.com/shen1994/README/raw/master/images/ChatRobot_predict.jpg)  
### 1.2 `python chat_robot.py`执行效果图  
![image](https://github.com/shen1994/README/raw/master/images/ChatRobot_chatchat.jpg)

## 2. 软件安装
 * recurrentshop下载地址: <https://github.com/farizrahman4u/recurrentshop>  
 * seq2seq 下载地址: <https://github.com/farizrahman4u/seq2seq>  
 * 微博数据(关于餐饮业,数据未清洗)下载地址
    私人地址: 链接: <https://pan.baidu.com/s/1g6l4_IDkLdLAjvrWf5sheQ> 密码: fxy3  

## 3. 参考链接
* seq2seq讲解: <http://jacoxu.com/encoder_decoder>  
* seq2seq数据读取: <http://suriyadeepan.github.io/2016-06-28-easy-seq2seq>  
* seq2seq论文地址: <https://arxiv.org/abs/1409.3215>  
* seq2seq+attention论文地址: <https://arxiv.org/abs/1409.0473>  
* ChatRobot启发论文: <https://arxiv.org/abs/1503.02364>  
* seq2seq源码: <https://github.com/farizrahman4u/seq2seq>  
* seq2seq源码需求: <https://github.com/farizrahman4u/recurrentshop>  
* beamsearch源码参考: <https://github.com/yanwii/seq2seq>  
* bucket源码参考: <https://github.com/1228337123/tensorflow-seq2seq-chatbot-zh>  

## 4. 系统介绍

* corpus文件夹  
1. `answer.txt` 问答语料-回答  
2. `question.txt` 问答语料-问题  

* data文件夹  
  本文件夹里的文件均由`data_process.py`生成，均为纯文本文件。  
  其中`dec_padding_ids.data`和`enc_padding_ids.data`将作为seq2seq网络的输入。  
>1. `dec_ids.data`  
>decode\_ids，将回答分词并转化为标号；  
>2. `dec_padding_ids.data`  
>经过padding后的dec\_ids，也就是将其中的每一句都补长到50个单词的长度；  
>3. `enc_ids.data`  
>encode\_ids，将问题分词并转化为标号；
>4. `enc_padding_ids.data`  
>经过padding后的enc\_ids，也就是将其中的每一句都补长到50个单词的长度；  

* model文件夹  
  本文件夹中存放了词典、模型权值等文件。  
>1. `dec_vocab20000.data`  
>回答的词典文件，通过该文件能够快速地将回答中的分词转成标号，或通过标号寻找对应的分词  
>2. `decoder_vector.m`  
>回答的word2vec文件，保存了分词对应的特征向量，由gensim库生成  
>3. `enc_vocab20000.data`  
>问题的词典文件，通过该文件能够快速地将问题中的分词转成标号，或通过标号寻找对应的分词  
>4. `encoder_vector.m`  
>问题的word2vec文件，保存了分词对应的特征向量，由gensim库生成   
>5. `seq2seq_model_weights.h5`  
>保存了训练好的seq2seq网络的权重

* recurrentshop文件夹  
  本文件夹包含了第三方库recurrentshop。  
  Recurrentshop是一个用于搭建复杂的循环神经网络的库，兼容Keras，使用Keras的标准实现了很多更复杂的RNN单元。  
>1. backend文件夹  
>由于Keras会使用不同的后端进行实现，因此针对两种不同的后端实现进行了针对性的实现与优化；  
>此外，通过该文件夹中的文件，也可以直接使用TensorFlow或是Theano使用recurrentshop库。
>2. `advanced_cells.py`  
>该文件实现了一个相对复杂的RNN的Cell，称为RHNCell，在我们的项目中没有使用。  
>3. `basic_cells.py`  
>该文件实现了recurrentshop中的三个基本RNNCell，分别是SimpleRNNCell，GRUCell，LSTMCell。  
>这是RNN中最常使用的三种单元，对应了使用tanh的RNN基本单元，GRU单元，LSTM单元。  
>recurrentshop将分别其实现并将其封装，使得这三种单元都可以被直接调用或是快速复用。  
>4. `engine.py`  
>为使得recurrentshop库中的三种RNN单元都能兼容Keras，因此这个库必须实现Keras中的所有方法；  
>对应的实现就在`engine.py`中。  
>5. `generic_utils.py`  
>一些序列化和反序列化的方法，让使用recurrentshop生成并训练的网络也能够通过Keras的接口存储到本地。

* seq2seq文件夹  
  本文件夹包含了第三方库seq2seq。  
  seq2seq是一个强大的序列到序列的训练库，使用recurrentshop作为基础，实现了各类seq2seq网络。  
>1. `cells.py`  
>实现了两种常用的解码器单元，LSTM解码器单元与Attention解码器单元；  
>其中Attention解码器就是在LSTM解码器单元上再加入一个Attention机制。  
>2. `models.py`  
>实现了三种常用的seq2seq网络，分别是SimpleSeq2Seq，Seq2Seq，AttentionSeq2Seq这三种网络。  
>其中SimpleSeq2Seq最简单，通过编码器将所有输入编码成一个特征向量，然后输入解码器；解码器只接受前一个解码器的输出作为自己的输入；  
>Seq2Seq可以选择与SimpleSeq2Seq一样的解码方式，也可以选择让每个解码器都接受[前一个解码器的输出，特征向量]作为自己的输入；  
>AttentionSeq2Seq可以选择Seq2Seq中的两种方式，也可以选择让每个解码器所接受的特征向量是不同的，作为“Attention”机制的体现。  

* word2cut文件夹  
  本文件夹包含了BiLSTM+CNN+CRF分词方法。  
>1. model文件夹  
>保存了模型的各项参数以及训练后的权重文件。  
>2. `bilstm_cnn_crf.py`  
>建立BiLSTM+CNN+CRF分词模型的主文件。
>3. `fake_keras.py`  
>由于使用了BiLSTM，因此使得所有句子变得等长是有必要的。  
>该文件中使用自带的pad\_sequences覆盖Keras的同名函数，使得我们可以自定义填充的一些细节。   
>这样我们就能通过pad\_sequences函数使得所有输入以自定义的方式被填充到等长。  
>4. `word_cut.py`  
>定义了Word\_cut类，封装了使用已训练的BiLSTM+CNN+CRF网络进行分词的各种方法  

* `chat_robot.py`  
  智能问答主程序，在各类准备工作完成后运行该文件就能够进行交互式问答。  

* `data_generator.py`  
  由于问答数据往往较少，因此我们需要使用数据生成器来生成、增强数据。  

* `data_process.py`
  封装了以上所述的分词数据处理方法，包括使用word2cut文件夹中的word\_cut类进行分词，对分词后的数据进行补长，生成问题和答案的词汇表。

* `encoder2decoder.py`  
  封装了使用gensim进行word2vec的功能，并能够返回word2vec矩阵，用于初始化Keras的Embedding层。  
  此外该文件也用于生成问答系统的网络结构，并使用word2vec矩阵初始化Embedding层，别的层的参数进行初始化后将整个模型返回。

* `score.py`  
  用于进行回答效果的评分（可选）。  

* `test.py`  
  用于进行单句问题的回答，同时也用于读取问答系统网络中除了Embedding层外别的层的数据参数并将其恢复。

* `train.py`  
  用于训练问答系统的网络。

* `word2vec.py`
  用于生成word2vec向量，使用Gensim库进行训练。

* `word2vec_plot.py`  
  用于生成的word2vec向量的可视化展示（可选）。  



## 5. 使用方法

 * 生成序列文件——将文字分词，转换成数字标号，补长到统一长度  
`python data_process.py`  

 * 生成word2vec向量，包括编码向量和解码向量  
`python word2vec.py`  

 * 训练网络  
`python train.py`  

 * 测试  
`python test.py`  

 * 模型评分（可选）  
`python score.py`  

 * 智能问答主程序  
`python chat_robot.py`  

 * 绘制word2vec向量分布图（可选）  
`python word2vec_plot.py`  

* 如果需要自定义语料集，请将question.txt和answer.txt放在corpus文件夹中，并重新运行上述所有命令（可选的可以不运行）

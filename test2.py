
from data_process import DataProcess
import jieba

if __name__ == "__main__":

    text = '南京市长江大桥'

    data_process = DataProcess(use_word2cut=True)

    words = data_process.text_cut_object.cut([text.strip()])
    print('BiLSTM-CNN-CRF:\t' + words[0])

    print('jieba:\t' + '/'.join(jieba.cut(text.strip())))

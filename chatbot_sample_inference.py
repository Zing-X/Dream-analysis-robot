import os
import numpy as np
import pandas as pd
#from chatbot_sample_training import tokenize_chinese, fit_sentence, build_model
from train_chatbot_GRU import tokenize_chinese, fit_sentence, build_model
#from train_chatbot_bert import tokenize_chinese, fit_sentence, build_model

### Control tensorflow won't occupied all your GPU memory
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
### Configs
MAXLEN = 256 ### MAXLEN should be same as the MAXLEN you use in previous training

### Load model and re-construct vocabulary you use in training
### Maybe you can save vocabulary used in training stage in advance
def load_model():
    voc = {} # Vocabulary dictionary
    voc_ind = 1 # vocabulary index start from 1, index 0 means nothing
    ### Import your training data for re-build vocabulary
    data = pd.read_csv('./data/data.csv')
    q = list(data['question'])[:50000]
    a = list(data['answer'])[:50000]

    ### Re-build vocabulary
    voc, voc_ind = tokenize_chinese(q, voc, voc_ind)
    voc, voc_ind = tokenize_chinese(a, voc, voc_ind)
    voc["<SOS>"] = len(voc)+1
    voc["<EOS>"] = len(voc)+1
    ind_voc = {}
    for k, v in voc.items():
        ind_voc[v] = k

    ### Build model and load the weight
    model = build_model(voc)
    model.load_weights('./models/chatbot_GRU.h5')

    return voc, ind_voc, model

### Predict single sentence and convert into format that can be send back to Line
def single_predict(input_text, voc, ind_voc, model):
    '''
        Inference data format:
            Question input:   [怎,麼,氣,死,接,我,程,式,的,學,弟,妹]     -->     [89, 3, 566, 677, 1258, 636, 88, 177, 33, 239, 464, 242, 0,...]
            t=0, Answer input: [<SOS>]             -->     [4977, 0, 0, 0,...]     -->   Expected prediction: [4977, 8, 0, 0,...]  
            t=1, Answer input: [<SOS>, 會]         -->     [4977, 8, 0, 0,...]     -->   Expected prediction: [4977, 8, 26, 0,...]
            t=2, Answer input: [<SOS>, 會, 有]     -->     [4977, 8, 26, 0,...]    -->   Expected prediction: [4977, 8, 26, 540,...]
             .                                                     .
             .                                                     .
             .                                                     .
            t=n, (End of prediction) --> Convert into human language: [8, 26, 540, 785, 4, 75, 23, 348, 12, 90, 33, 58, 4978, 0,... ] --> [會,有,報,應,' ',還,是,別,這,樣,的,好,<EOS>]
    '''
    ### Answer input initial state --> [<SOS>] --> [4977, 0, 0, 0,...]
    ans_input = np.zeros((1,MAXLEN),dtype='int64')
    ans_input[0,0] = voc["<SOS>"]

    ### Convert your question into number label
    res = fit_sentence(input_text, voc)
    while len(res) < MAXLEN: # If sentence is shorter than maxlen, append 0 until length reach maxlen
        res.append(0)
    question = np.array([res])

    ind = 0
    ### Stop condition: 1. Prediction until MAXLEN, 2. Output '<EOS>' or 0
    while (ans_input[0][ind] != voc['<EOS>'] or ans_input[0][ind] != 0) and ind < MAXLEN-1:
        ind += 1
        pred = model.predict([question, ans_input])
        res = np.argmax(pred,axis=-1) # Get highest probabilty index of character in vocabulary
        ans_input[0,ind] = res[0][ind]
    ### Convert prediction into human language 
    ans = ""
    for i in ans_input[0][1:]:
        if i == 0 or i == voc['<EOS>']:
            break
        ans += ind_voc[i] 
    return str(ans) # Return string --> Send string back to ngrok and forwarding to Line

if __name__ == '__main__':
    ### Load vocabulary and model
    voc, ind_voc, model = load_model()

    ### Asking for question forever until the end
    while True:
        question = input('請說: ')
        if question == '88': ### Input '88' means terminate the chatbot
            print('再見')
            break
        elif question == '亂講':
            print('對不起，我是一隻笨猴子')
        else:
            res = single_predict(question, voc, ind_voc, model) 
            print(res)
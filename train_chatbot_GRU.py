import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, GRU, Dense, Embedding
from tensorflow.keras.models import Model


### Control tensorflow won't occupied all your GPU memory
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
### Configs --> Parameters you can attempt to tuning
MAXLEN = 256 # If you data has short sentence, please try lower MAXLEN in order to increase performance of model training
EPOCHS = 200
BATCH_SIZE = 64
EMB_DIM = 100
UNIT = 64

### Build character-based vocabulary
def tokenize_chinese(texts, voc, voc_ind):
    for t in tqdm(texts):
        for ch in str(t):
            if ch not in voc:
                voc[ch] = voc_ind
                voc_ind += 1 
    return voc, voc_ind

### Convert human language sentence into vocabulary numbers
### Ex. [I, am, a, dog] --> [226, 543, 78, 98]
def fit_sentence(sen, voc):
    res = []
    for i in sen:
        res.append(voc[i])
    return res

### Build chatbot model
def build_model(voc):
    Q_in = Input((MAXLEN,), name='Q_input')
    Q_emb = Embedding(len(voc) + 1, EMB_DIM, mask_zero=True, name='Q_emb')(Q_in)
    _, Q_h = GRU(UNIT, return_sequences=True, return_state=True, recurrent_dropout=0.2, name='Q_GRU')(Q_emb)

    A_in = Input((MAXLEN,), name='A_input')
    A_emb = Embedding(len(voc) + 1, EMB_DIM, mask_zero=True, name='A_emb')(A_in)
    A_out = GRU(UNIT, return_sequences=True, recurrent_dropout=0.2, name='A_GRU')(A_emb, initial_state=Q_h)

    output = Dense(len(voc) + 1, activation='softmax', name='Output')(A_out)

    model = Model(inputs=[Q_in, A_in], outputs=output, name='Gossip_ChatBot')

    return model

if __name__ == '__main__':

    voc = {} # Vocabulary dictionary
    voc_ind = 1 # vocabulary index start from 1, index 0 means nothing
    ''' Import your data
        Your data should be like:
            Question: 怎麼氣死接我程式的學弟妹
            Answer: 會有報應 還是別這樣的好
    '''
    ### Please change data into your own data
    data = pd.read_csv('./data/data.csv')
    q = list(data['question'])[:50000]
    a = list(data['answer'])[:50000]

    ### Build vocabulary --> Evaluate how many characters in your data
    voc, voc_ind = tokenize_chinese(q, voc, voc_ind)
    voc, voc_ind = tokenize_chinese(a, voc, voc_ind)
    
    ### Insert "Start Of Sentence" token into vocabulary
    voc["<SOS>"] = len(voc)+1
    ### Insert "End Of Sentence" token into vocabulary
    voc["<EOS>"] = len(voc)+1

    ### Convert your sentence into number label and add "<SOS>" at front, "<EOS>" at end
    '''
        Training data format:
            q_x:   [怎,麼,氣,死,接,我,程,式,的,學,弟,妹]                -->     [89, 3, 566, 677, 1258, 636, 88, 177, 33, 239, 464, 242, 0,...]
            ans_x: [<SOS>,會,有,報,應,' ',還,是,別,這,樣,的,好,<EOS>]   -->     [4977, 8, 26, 540, 785, 4, 75, 23, 348, 12, 90, 33, 58, 4978, 0,...]
            ans_y: [會,有,報,應,' ',還,是,別,這,樣,的,好,<EOS>]         -->     [8, 26, 540, 785, 4, 75, 23, 348, 12, 90, 33, 58, 4978, 0,... ]
        Human language will convert into number label, mapping from vocabulary we build above
    '''
    q_x = []
    ans_x, ans_y = [], []
    ### Question input
    for i in tqdm(q):
        res = fit_sentence(i, voc)
        while len(res) < MAXLEN: ### If sentence is shorter than maxlen, append 0 until length reach maxlen
            res.append(0)
        q_x.append(res)
    ### Answer input
    for i in tqdm(a):
        res = fit_sentence(i, voc)
        res.insert(0,voc["<SOS>"])
        res.append(voc["<EOS>"])
        while len(res) < MAXLEN: ### If sentence is shorter than maxlen, append 0 until length reach maxlen
            res.append(0)
        ans_x.append(res)
    ### Answer output
    for i in ans_x:
        tmp = i[1:]
        tmp.append(0)
        ans_y.append(tmp)
    
    ### Turn into np.array for training
    q_x = np.array(q_x)
    ans_x = np.array(ans_x)
    ans_y = np.array(ans_y)

    ### Build model and compile model
    model = build_model(voc)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    ### Callbacks --> Checkpoint: Change file path to the directory where you want to save your model
    checkpoint = ModelCheckpoint(filepath="./models/chatbot_GRU.h5", monitor='accuracy',verbose=1,save_best_only=True,save_weights_only=True)
    ### Callbacks --> Earlystop: Monitor accuracy and decide whether to stop the training procedure
    earlystop = EarlyStopping(monitor='accuracy',patience=10,verbose=1)

    ### If you have model trained before, you can load it back and continue previous training procedure
    try:
        model.load_weights('./models/chatbot_GRU.h5')
        print("Load model...")
    ### If you haven't train any model yet, train model from initial
    except:
        print("Fail to load model...")

    ### Train your model
    model.fit((q_x, ans_x), ans_y, batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[checkpoint, earlystop],verbose=1)


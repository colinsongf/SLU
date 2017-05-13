import numpy as np
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.core import Lambda, Flatten
from keras.layers import Activation, Dense, Input
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from gendata import gendata

from RecurrentNeuralNetwork import conlleval, get_perf

dataset = gendata()
trainSentences, trainY, trainL = dataset['train']
validSentences, validY, validL = dataset['valid']
testSentences,  testY, testL  = dataset['test']
idx2labels = dataset['idx2labels']
idx2words = dataset['idx2words']
idx2intents = dataset['idx2intents']

lengths = [len(x) for x in trainSentences]
print 'Input sequence length range: ', max(lengths), min(lengths)

maxlen = max(lengths)
print 'Maximum sequence length:', maxlen

from keras.utils.np_utils import to_categorical

X_train = pad_sequences(trainSentences, maxlen=maxlen)
y_train = pad_sequences(trainY, maxlen=maxlen)
y_train = np.array([to_categorical(x, num_classes=len(idx2labels)) for x in y_train])
l_train = np.array(to_categorical(trainL, num_classes=len(idx2intents)))

X_valid = pad_sequences(validSentences, maxlen=maxlen)
y_valid = pad_sequences(validY, maxlen=maxlen)
y_valid = np.array([to_categorical(x, num_classes=len(idx2labels)) for x in y_valid])
l_valid = np.array(to_categorical(validL, num_classes=len(idx2intents)))

X_test = pad_sequences(testSentences, maxlen=maxlen)
y_test = pad_sequences(testY, maxlen=maxlen)
y_test = np.array([to_categorical(x, num_classes=len(idx2labels)) for x in y_test])


from keras.constraints import maxnorm
from keras import regularizers
from keras.layers.merge import Concatenate
vocab_size = len(idx2words)
embedding_size = 300
hidden_size = 128
slot_size = len(idx2labels)
intent_size = len(idx2intents)

main_input = Input(shape=(maxlen,), dtype='int32', name='main_input')
embed = Embedding(vocab_size, embedding_size, input_length=maxlen, mask_zero=True)(main_input)
tagembed = Input(shape=(maxlen, slot_size), name='tag_input')

merge = Concatenate(axis=-1)([embed, tagembed])
lstm = Bidirectional(LSTM(hidden_size, return_sequences=True, kernel_constraint=maxnorm(5.)), merge_mode='concat')(embed)
#lstm = LSTM(hidden_size, return_sequences=True, kernel_constraint=maxnorm(3.))(merge)

slots = TimeDistributed(Dense(slot_size, kernel_constraint=maxnorm(5.), kernel_regularizer=regularizers.l2(0.0001)))(lstm)
outputslots = Activation('softmax', name='so')(slots)

sliced = Lambda(lambda x : x[:, -1, :])(lstm)
intents = Dense(intent_size, kernel_constraint=maxnorm(5.), kernel_regularizer=regularizers.l2(0.0001))(sliced)
outputintents = Activation('softmax', name='io')(intents)

model = Model(inputs=[main_input], outputs = [outputslots])
#model2 = Model(inputs=[main_input, tagembed], outputs = [outputintents])

print "compiling"
model.compile(loss='categorical_crossentropy', optimizer='adam')
#model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics={'io' : 'accuracy'})

batch_size = 32

print "fitting"

def remove_padding(pred, actu) :
    output = []
    for a,b in zip(pred, actu) :
        if b != 0 :
            output.append(a)

    return output

wordsValid = [map(lambda x: idx2words[x], w) for w in validSentences]
validAct = [map(lambda x: idx2labels[x], y) for y in validY]
wordsTest = [map(lambda x: idx2words[x], w) for w in testSentences]
testAct = [map(lambda x: idx2labels[x], y) for y in testY]

def predict_classes(x) :
    pred = model.predict(x)
    return pred.argmax(axis=-1)

while(True) :
    model.fit([X_train], [y_train], batch_size=batch_size, nb_epoch=1, validation_data=([X_valid], [y_valid]))
    validPred = [map(lambda x: idx2labels[x], remove_padding(c, d)) for c, d in zip(predict_classes([X_valid]), X_valid)]
    print conlleval(validPred, validAct, wordsValid, 'output/current.valid.txt')
    testPred = [map(lambda x: idx2labels[x], remove_padding(c, d)) for c, d in zip(predict_classes([X_test]), X_test)]
    print conlleval(testPred, testAct, wordsTest, 'output/current.test.txt')

# while(True) :
#     model2.fit([X_train, y_train], [l_train], batch_size=batch_size, nb_epoch=1, validation_data=([X_valid, y_valid], [l_valid]))
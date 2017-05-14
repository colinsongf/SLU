import numpy as np
from keras.models import Model

from keras.layers.recurrent import LSTM
from keras.layers.core import Lambda, Flatten
from keras.layers import Activation, Dense, Input
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate

from keras.constraints import maxnorm
from keras import regularizers

from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from gendata import gendata

from utils import conlleval, get_perf

if __name__ == "__main__" :
    import sys

    args = sys.argv

    if len(args) > 1 :
        input_dir = "/input/"
        output_dir = "/output/"
    else :
        input_dir = "input/"
        output_dir = "output/"

    dataset = gendata(input_dir + "ATIS_samples/")
    trainSentences, trainY, trainL = dataset['train']
    validSentences, validY, validL = dataset['valid']
    testSentences,  testY, testL  = dataset['test']
    idx2labels = dataset['idx2labels']
    idx2words = dataset['idx2words']
    idx2intents = dataset['idx2intents']

    trainSentences += validSentences
    trainY += validY
    trainL += validL

    lengths = [len(x) for x in trainSentences]
    print 'Input sequence length range: ', max(lengths), min(lengths)

    maxlen = max(lengths)
    print 'Maximum sequence length:', maxlen

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
    l_test = np.array(to_categorical(testL, num_classes=len(idx2intents)))

    vocab_size = len(idx2words)
    embedding_size = 300
    hidden_size = 128
    slot_size = len(idx2labels)
    intent_size = len(idx2intents)

    main_input = Input(shape=(maxlen,), dtype='int32', name='main_input')
    embed = Embedding(vocab_size, embedding_size, input_length=maxlen, mask_zero=True)(main_input)

    lstm_slots = Bidirectional(LSTM(hidden_size, return_sequences=True, kernel_constraint=maxnorm(3.)), merge_mode='sum')(embed)
    lstm_intent = LSTM(hidden_size, return_sequences=True, kernel_constraint=maxnorm(3.))(embed)

    slots = TimeDistributed(Dense(slot_size, kernel_constraint=maxnorm(3.), kernel_regularizer=regularizers.l2(0.0001)))(lstm_slots)
    outputslots = Activation('softmax', name='so')(slots)

    sliced = Lambda(lambda x : x[:, -1, :])(lstm_intent)
    intents = Dense(intent_size, kernel_constraint=maxnorm(3.), kernel_regularizer=regularizers.l2(0.0001))(sliced)
    outputintents = Activation('softmax', name='io')(intents)

    model_slots = Model(inputs=[main_input], outputs = [outputslots])
    model_intent = Model(inputs=[main_input], outputs = [outputintents])

    print "compiling"
    model_slots.compile(loss='categorical_crossentropy', optimizer='adam')
    model_intent.compile(loss='categorical_crossentropy', optimizer='adam', metrics={'io' : 'accuracy'})

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

    # from keras import backend as K
    # def ranking_loss(output, target) :
    #   return K.log(1. + K.exp(K.maximum(output))) + K.log(1. + K.exp(K.sum(output * target)))

    def predict_classes(pred) :
        return pred.argmax(axis=-1)


    i = 0
    while(True) :
        model_intent.fit([X_train], [l_train], batch_size=batch_size, epochs=1, validation_data=([X_test], [l_test]))
        model_slots.fit([X_train], [y_train], batch_size=batch_size, epochs=1, validation_data=([X_valid], [y_valid]))

        validPred = predict_classes(model_slots.predict([X_valid]))
        validPred = [map(lambda x: idx2labels[x], remove_padding(c, d)) for c, d in zip(validPred, X_valid)]
        print conlleval(validPred, validAct, wordsValid, output_dir + 'current.valid.txt')

        testPred = predict_classes(model_slots.predict([X_test]))
        testPred = [map(lambda x: idx2labels[x], remove_padding(c, d)) for c, d in zip(testPred, X_test)]
        print conlleval(testPred, testAct, wordsTest, output_dir + 'current.test.txt')

        model_slots.save(output_dir + 'model_slots_' + str(i) + '.ckpt')
        i = 1 - i
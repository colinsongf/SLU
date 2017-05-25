import numpy as np
from keras import backend as K
from keras.models import Model, load_model

from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Lambda, Flatten, Dropout
from keras.layers import Activation, Dense, Input
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate, Dot
from keras.layers.normalization import BatchNormalization

from keras.constraints import maxnorm
from keras import regularizers
from keras import metrics
from keras import optimizers

from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from gendata import gendata, map_sentence

from utils import conlleval, get_perf, accuracy


def predict_classes(pred) :
        return pred.argmax(axis=-1)

def acc_2(y_true, y_pred) :
    return metrics.top_k_categorical_accuracy(y_true, y_pred,k=1)

import sys

args = sys.argv

if 'floyd' in args :
    input_dir = "/input/"
    output_dir = "/output/"
else :
    input_dir = "/media/sarthak/Data/MAJOR/Major/rnnsimple/input/"
    output_dir = "/media/sarthak/Data/MAJOR/Major/rnnsimple/output/"

dataset = gendata(input_dir + "ATIS_samples/")
trainSentences, trainY, trainL, trainlist = dataset['train']
validSentences, validY, validL, validlist = dataset['valid']
testSentences,  testY, testL, testlist  = dataset['test']
idx2labels = dataset['idx2labels']
idx2words = dataset['idx2words']
idx2intents = dataset['idx2intents']

lengths = [len(x) for x in trainSentences]
print 'Input sequence length range: ', max(lengths), min(lengths)

maxlen = max(lengths)
print 'Maximum sequence length:', maxlen

X_train = pad_sequences(trainSentences, maxlen=maxlen)
y_train = pad_sequences(trainY, maxlen=maxlen)
y_train = np.array([to_categorical(x, num_classes=len(idx2labels)) for x in y_train])
l_train = np.array(to_categorical(trainL, num_classes=len(idx2intents)))

list_train = pad_sequences(trainlist, maxlen=maxlen)
list_train = np.array([to_categorical(x, num_classes=len(idx2intents)) for x in list_train])

X_valid = pad_sequences(validSentences, maxlen=maxlen)
y_valid = pad_sequences(validY, maxlen=maxlen)
y_valid = np.array([to_categorical(x, num_classes=len(idx2labels)) for x in y_valid])
l_valid = np.array(to_categorical(validL, num_classes=len(idx2intents)))

list_valid = pad_sequences(validlist, maxlen=maxlen)
list_valid = np.array([to_categorical(x, num_classes=len(idx2intents)) for x in list_valid])

X_test = pad_sequences(testSentences, maxlen=maxlen)
y_test = pad_sequences(testY, maxlen=maxlen)
y_test = np.array([to_categorical(x, num_classes=len(idx2labels)) for x in y_test])
l_test = np.array(to_categorical(testL, num_classes=len(idx2intents)))

list_test = pad_sequences(testlist, maxlen=maxlen)
list_test = np.array([to_categorical(x, num_classes=len(idx2intents)) for x in list_test])

vocab_size = len(idx2words)
embedding_size = 300
hidden_size = 150
slot_size = len(idx2labels)
intent_size = len(idx2intents)


wordsValid = [map(lambda x: idx2words[x], w) for w in validSentences]
validAct = [map(lambda x: idx2labels[x], y) for y in validY]
wordsTest = [map(lambda x: idx2words[x], w) for w in testSentences]
testAct = [map(lambda x: idx2labels[x], y) for y in testY]

def remove_padding(pred, actu) :
    output = []
    for a,b in zip(pred, actu) :
        if b != 0 :
            output.append(a)

    return output

if __name__ == "__main__" :

    batch_size = 32

    print "fitting"

    if 'test' not in args :
        main_input = Input(shape=(maxlen,), dtype='int32', name='main_input')
        embed = Embedding(vocab_size, embedding_size, input_length=maxlen, mask_zero=True)(main_input)

        lstm_slots = Bidirectional(GRU(hidden_size, dropout=0.5, recurrent_dropout=0.4,
                                    return_sequences=True, kernel_constraint=maxnorm(5.)), merge_mode='sum')(embed)
        td_slots = TimeDistributed(Dense(slot_size, kernel_constraint=maxnorm(5.0), kernel_regularizer=regularizers.l2(0.0001)))(lstm_slots)

        outputslots = Activation('softmax', name='so')(BatchNormalization()(td_slots))

        lstm_intents = Bidirectional(LSTM(hidden_size, dropout=0.4, recurrent_dropout=0.4,
                                        return_sequences=True, kernel_constraint=maxnorm(5.)), merge_mode='sum')(embed)
        td_intents = TimeDistributed(Dense(intent_size, kernel_constraint=maxnorm(5.0), kernel_regularizer=regularizers.l2(0.0001)))(lstm_slots)

        output_inter_intents = Activation('softmax', name='so_in')(td_intents)

        sum_intents = Lambda(lambda x : K.sum(x, axis=1, keepdims=False))(embed)
        intents = Dense(intent_size, kernel_regularizer=regularizers.l2(0.0001))(sum_intents)
        outputintents = Activation('softmax', name='io')(BatchNormalization()(intents))

        model_slots = Model(inputs=[main_input], outputs = [outputslots, outputintents])

        print "compiling"
        adam = optimizers.adam(clipnorm=5.)

        model_slots.compile(loss='categorical_crossentropy', optimizer=adam, metrics={'io' : acc_2}, loss_weights={'io' : 0.5, 'so' : 1.0})

        i = 0
        while(True) :
            model_slots.fit([X_train], [y_train, l_train], batch_size=batch_size, epochs=1,
                            validation_data=([X_valid], [y_valid, l_valid]))
            model_slots.save(output_dir + 'model_slots_intents_' + str(i) + '.ckpt')

            valid_predicts = model_slots.predict([X_valid])
            validPred = predict_classes(valid_predicts[0])
            validPred = [map(lambda x: idx2labels[x], remove_padding(c, d)) for c, d in zip(validPred, X_valid)]
            print conlleval(validPred, validAct, wordsValid, output_dir + 'current.valid.txt')

            print "Valid Accuracy : " + str(accuracy(predict_classes(valid_predicts[1]), validL))

            test_predicts = model_slots.predict([X_test])
            testPred = predict_classes(test_predicts[0])
            testPred = [map(lambda x: idx2labels[x], remove_padding(c, d)) for c, d in zip(testPred, X_test)]
            print conlleval(testPred, testAct, wordsTest, output_dir + 'current.test.txt')

            print "Test Accuracy : " + str(accuracy(predict_classes(test_predicts[1]), testL))


            i = 1 - i


import record
print "Loading Model ... "
import script1
print "DT Parser loaded"
model_intent = load_model(output_dir + 'model_intent_1.ckpt', custom_objects={'acc_2':acc_2})#
model_slots = load_model(output_dir + 'model_slots_1.ckpt')
print "Model Loaded"
thresh = record.find_threshold()

def test(sentence) :
    sentence, actual_sentence = map_sentence(sentence, input_dir + 'ATIS_samples/')
    X_sent = pad_sequences([sentence], maxlen=maxlen)
    sentPred = predict_classes(model_slots.predict([X_sent]))
    sentPred = [map(lambda x: idx2labels[x], remove_padding(c, d)) for c, d in zip(sentPred, X_sent)]

    intentPred = predict_classes(model_intent.predict([X_sent])[0])
    intentPred = [idx2intents[c] for c in intentPred]
    return { "tags" : sentPred[0], "intent" : intentPred[0], "quote_output" : script1.get_quotes([actual_sentence], sentPred) }

def runtest() :
    test_predicts = model_slots.predict([X_test])
    testPred = predict_classes(test_predicts)
    testPred = [map(lambda x: idx2labels[x], remove_padding(c, d)) for c, d in zip(testPred, X_test)]
    print conlleval(testPred, testAct, wordsTest, output_dir + 'current.test.txt')

    intentPred = predict_classes(model_intent.predict([X_test])[0])
    print "Test Accuracy : " + str(accuracy(intentPred, testL))

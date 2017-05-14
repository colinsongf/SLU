
def getids(data_dir) :

    f = open(data_dir + 'out_vocab_123.txt').readlines()
    output = {f[i].strip():i for i in range(len(f))}

    n = 'train'
    e = '.seq.out'
    fp = open(data_dir + n + '/' + n + e).readlines()
    fp = [x.strip() for x in fp]
    fp = [x.split(' ')[1:] for x in fp]

    print output
    tagids = []
    for s in fp :
        tokens = []
        for w in s :
            if w in output :
                tokens.append(output[w])
            else :
                tokens.append(output['_UNK'])

        tagids.append(tokens)

    print tagids[0]

def gendata(data_dir) :
    f = ['train', 'test', 'valid']
    ext = ['.ids10000.seq.in', '.ids123.seq.out']
    output = {k:[] for k in f}
    for n in f :
        for e in ext :
            fp = open(data_dir + n + '/' + n + e).readlines()
            fp = [x.strip() for x in fp]
            fp = [x.split(' ') for x in fp]
            fp = [[int(a) for a in x] for x in fp]
            fp = [x[1:] for x in fp]

            output[n].append(fp)

        e = '.ids.labels'
        fp = open(data_dir + n + '/' + n + e).readlines()
        fp = [x.strip() for x in fp]
        fp = [int(x) for x in fp]
        output[n].append(fp)

        output[n] = tuple(output[n])

    f = open(data_dir + 'in_vocab_10000.txt').readlines()
    output['idx2words'] = {i:f[i].strip() for i in range(len(f))}

    f = open(data_dir + 'out_vocab_123.txt').readlines()
    output['idx2labels'] = {i:f[i].strip() for i in range(len(f))}

    f = open(data_dir + 'labels.txt').readlines()
    output['idx2intents'] = {i:f[i].strip() for i in range(len(f))}
    return output

def map_sentence(sentence, data_dir) :
    f = open(data_dir + 'in_vocab_10000.txt').readlines()
    output = {f[i].strip():i for i in range(len(f))}

    sentence = sentence.strip().split(' ')
    tokens = []
    for s in sentence :
        if s in output :
            tokens.append(output[s])
        else :
            tokens.append(output['_UNK'])

    return tokens

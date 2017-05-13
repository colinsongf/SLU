
def gendata() :
    data_dir = 'input/ATIS_samples/'
    f = ['train', 'test', 'valid']
    ext = ['.ids10000.seq.in', '.ids123.seq.out', '.ids.labels']
    output = {k:[] for k in f}
    for n in f :
        for e in ext :
            fp = open(data_dir + n + '/' + n + e).readlines()
            fp = [x.strip() for x in fp]
            if 'labels' not in e :
                fp = [x.split(' ') for x in fp]
                fp = [[int(a) for a in x] for x in fp]
                fp = [x[1:] for x in fp]
            else :
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


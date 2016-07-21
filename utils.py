import subprocess

def conlleval(p, g, w, filename):
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename)


def get_perf(filename):
    _conlleval = 'conlleval.pl'
    proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    print out
    precision = float(out[3][:-2])
    recall = float(out[5][:-2])
    f1score = float(out[7])

    return {'p': precision, 'r': recall, 'f1': f1score}

def accuracy(y_true, y_pred) :
    n = len(y_true)
    assert len(y_true) == len(y_pred)
    eq = 0.0
    for a, b in zip(y_true, y_pred) :
        if a == b :
            eq += 1.0

    return eq/n



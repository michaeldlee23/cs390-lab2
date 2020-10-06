import os, sys
import matplotlib.pyplot as plt
from collections import OrderedDict


def summarize(algorithm, accuracies, outfile):
    accuracies = OrderedDict(sorted(accuracies.items(), key=lambda x: x[0]))
    plt.bar(list(accuracies.keys()), list(accuracies.values()), color='cyan', width=0.5)
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of %s' % algorithm)
    plt.tight_layout()

    labels = [str(x) for x in accuracies.values()]
    for i in range(len(labels)):
        plt.annotate(labels[i] + '%', xy=(i, float(labels[i])), ha='center', va='bottom')
    print('Saving accuracy plot to %s...' % outfile)
    plt.savefig(outfile)


# Format of beginning of meta.txt:
# tf_net
# cifar_100_f
# epochs=20
# batchSize=100
# dropRate=0.25
# accuracy=19.300000%
def parseData(filename):
    algorithm = ''
    accuracies = dict()
    for item in os.listdir(filename):
        metaPath = os.path.join(filename, item + '/meta.txt')
        meta = open(metaPath, 'r')
        lines = meta.readlines()
        algorithm = lines[0]
        accuracies[lines[1]] = float(lines[5][9:-2])
    outfile = './assets/' + filename[filename.rfind('/') + 1:] + '-accuracy-plot.png'
    return algorithm, accuracies, outfile


def main():
    argv = sys.argv[1:]
    algorithm, accuracies, outfile = parseData(argv[0])
    summarize(algorithm, accuracies, outfile)


if __name__ == '__main__':
    main()

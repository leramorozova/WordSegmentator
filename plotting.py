import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# гарфик качества обучении при каждом новом разбиении на блоки

def ccv(x, y1, y2):
    plt.plot(x, y1, color='red', linewidth=4)
    plt.xlim(0, 0.99)
    plt.title('Accuracy', fontsize='x-large')
    plt.xlabel('ITERATION')
    plt.ylabel('SCORE')
    plt.savefig('./plots/total_accuracy_bl.png', format='png', dpi=100)
    plt.clf()
    plt.plot(x, y2, color='red', linewidth=4)
    plt.xlim(0, 0.99)
    plt.title('Sentence_match', fontsize='x-large')
    plt.xlabel('ITERATION')
    plt.ylabel('SCORE')
    plt.savefig('./plots/total_smatch_bl.png', format='png', dpi=100)
    
# графики по гиперпараметрам

def epochs(acc, sm, x):
    plt.plot(x, acc, color='red', linewidth=4)
    plt.ylim(0, 0.99)
    plt.xlabel('AMOUNT OF TRAINING CYCLES', fontsize='x-large')
    plt.ylabel('ACCURACY')
    plt.savefig('./plots/epochs_accuracy.png', format='png', dpi=100)
    plt.clf()
    plt.plot(x, sm, color='red', linewidth=4)
    plt.ylim(0, 0.99)
    plt.xlabel('AMOUNT OF TRAINING CYCLES', fontsize='x-large')
    plt.ylabel('SENTENCE MATCH')
    plt.savefig('./plots/epochs_smatch.png', format='png', dpi=100)

def hidden_nodes(acc, sm, x):
    plt.plot(x, acc, color='red', linewidth=4)
    plt.xlabel('AMOUNT OF HIDDEN NODES', fontsize='x-large')
    plt.ylabel('ACCURACY')
    plt.savefig('./plots/hn_accuracy.png', format='png', dpi=100)
    plt.clf()
    plt.plot(x, sm, color='red', linewidth=4)
    plt.xlabel('AMOUNT OF HIDDEN NODES', fontsize='x-large')
    plt.ylabel('SENTENCE MATCH')
    plt.savefig('./plots/hn_smatch.png', format='png', dpi=100)

def learning_rate(acc, sm, x):
    plt.plot(x, acc, color='red', linewidth=4)
    plt.ylim(0, 0.99)
    plt.xlabel('LEARNING RATE', fontsize='x-large')
    plt.ylabel('ACCURACY')
    plt.savefig('./plots/lr_accuracy.png', format='png', dpi=100)
    plt.clf()
    plt.plot(x, sm, color='red', linewidth=4)
    plt.ylim(0, 0.99)
    plt.xlabel('LEARNING RATE', fontsize='x-large')
    plt.ylabel('SENTENCE MATCH')
    plt.savefig('./plots/lr_smatch.png', format='png', dpi=100)
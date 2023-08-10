import matplotlib.pyplot as plt
import os
acc = []
loss = []
tacc = []
path01 = 'outputs/lr_test/run.log'
ly=[]
def plot_result():
    getly()
    '''
    plt.subplot(331)
    plt.plot(ly,get5(acc[0:200]), label="lr=0.001 train")
    plt.plot(ly,get5(tacc[0:200]), label="lr=0.001 test")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title('lr=0.001')
    plt.subplot(332)
    plt.plot(ly,get5(acc[200:400]), label="lr=0.0001 train")
    plt.plot(ly,get5(tacc[200:400]), label="lr=0.0001 test")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title('lr=0.0001')
    plt.subplot(333)
    plt.plot(ly,get5(acc[400:600]), label="lr=0.0005 train")
    plt.plot(ly,get5(tacc[400:600]), label="lr=0.0005 test")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title('lr=0.0005')
    plt.subplot(334)
    plt.plot(ly,get5(acc[600:800]), label="lr=0.00001 train")
    plt.plot(ly,get5(tacc[600:800]), label="lr=0.00001 test")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title('lr=0.00001')
    plt.subplot(335)
    plt.plot(ly,get5(acc[800:1000]), label="lr=0.00005 train")
    plt.plot(ly,get5(tacc[800:1000]), label="lr=0.00005 test")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title('lr=0.00005')
    plt.subplot(336)
    plt.plot(ly,get5(acc[1000:1200]), label="lr=0.000001 train")
    plt.plot(ly,get5(tacc[1000:1200]), label="lr=0.000001 test")
    plt.title('lr=0.000001')
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    plt.subplot(337)
    plt.plot(ly,get5(acc[1200:1400]), label="lr=0.000005 train")
    plt.plot(ly,get5(tacc[1200:1400]), label="lr=0.000005 test")
    plt.title('lr=0.000005')
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()


    #plt.plot(loss, label="loss")
    #plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
        wspace=None, hspace=0.7)
    plt.show()
    '''

    plt.plot(ly,get5(acc[0:200]), label="lr=0.001")
    plt.plot(ly,get5(acc[200:400]), label="lr=0.0001")
    plt.plot(ly,get5(acc[400:600]), label="lr=0.0005")
    plt.plot(ly,get5(acc[600:800]), label="lr=0.00001")
    plt.plot(ly,get5(acc[800:1000]), label="lr=0.00005")
    plt.plot(ly,get5(acc[1000:1200]), label="lr=0.000001")
    plt.plot(ly,get5(acc[1200:1400]), label="lr=0.000005")
    plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

def readlog(path,tacc,acc,loss):
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            if 'train acc' in line:
                train_loss = float(line.split(':')[1].split(',')[0].strip())
                train_acc = float(line.split(':')[2].split(',')[0].strip())
                print(train_acc)
                loss.append(train_loss)
                acc.append(train_acc)
            if 'test acc' in line:
                test_acc = float(line.split(':')[2].split(',')[0].strip())
                tacc.append(test_acc)
            line = f.readline()

def get5(ls):
    n = -1
    lsnew = []
    for i in ls:
        n = n + 1
        if n % 20 == 0 or n == 199:
            lsnew.append(i)
            print(n)
    return lsnew

def getly():
    for i in range(40):
        if i % 4 == 0 or i == 39:
            ly.append(i)
    print(ly)
readlog(path01,tacc,acc,loss)
plt.figure(figsize=(10,5))
plot_result()

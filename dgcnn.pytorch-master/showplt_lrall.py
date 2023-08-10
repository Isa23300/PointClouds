import matplotlib.pyplot as plt
import os
acc1 = []
acc2 = []
acc3 = []
acc4 = []
acc5 = []
acc6 = []
acc7 = []
acc11 = []
acc22 = []
acc33 = []
acc44 = []
acc55 = []
acc66 = []
acc77 = []
loss11 = []
loss22 = []
loss33 = []
loss44 = []
loss55 = []
loss66 = []
loss77 = []
loss1 = []
loss2 = []
loss3 = []
loss4 = []
loss5 = []
loss6 = []
loss7 = []


path1 = 'outputs/lr2023-08-04_08-23_/dgcnn.txt'
path2 = 'outputs/lr2023-08-04_10-07_/dgcnn.txt'
path3 = 'outputs/lr_0.0005_2023-08-08_10-44_/dgcnn.txt'
path4 = 'outputs/lr2023-08-04_13-31_/dgcnn.txt'
path5 = 'outputs/lr_5e-05_2023-08-08_12-23_/dgcnn.txt'
path6 = 'outputs/lr2023-08-04_16-58_/dgcnn.txt'
path7 = 'outputs/lr2023-08-04_18-40_/dgcnn.txt'


def plot5_resultacc():
    '''
    plt.subplot(331)
    plt.plot(get5y(acc1),get5(acc1), label="lr=0.001 train")
    plt.plot(get5y(acc11),get5(acc11), label="lr=0.001 validation")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title('lr=0.001')
    plt.subplot(332)
    plt.plot(get5y(acc2),get5(acc2), label="lr=0.0001 train")
    plt.plot(get5y(acc22),get5(acc22), label="lr=0.0001 validation")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title('lr=0.0001')
    plt.subplot(333)
    plt.plot(get5y(acc3),get5(acc3), label="lr=0.0005 train")
    plt.plot(get5y(acc33),get5(acc33), label="lr=0.0005 validation")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title('lr=0.00015')
    plt.subplot(334)
    plt.plot(get5y(acc4),get5(acc4), label="lr=0.00001 train")
    plt.plot(get5y(acc44),get5(acc44), label="lr=0.00001 validation")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title('lr=0.00001')
    plt.subplot(335)
    plt.plot(get5y(acc5),get5(acc5), label="lr=0.00005 train")
    plt.plot(get5y(acc55),get5(acc55), label="lr=0.00005 validation")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title('lr=0.000015')
    plt.subplot(336)
    plt.plot(get5y(acc6),get5(acc6), label="lr=0.000001 train")
    plt.plot(get5y(acc66),get5(acc66), label="lr=0.000001 validation")
    plt.title('lr=0.000001')
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    plt.subplot(337)
    plt.plot(get5y(acc7),get5(acc7), label="lr=0.000005 train")
    plt.plot(get5y(acc77),get5(acc77), label="lr=0.000005 validation")
    plt.title('lr=0.000005')
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    '''
    plt.plot(get5y(acc11),get5(acc11), label="lr=0.001")
    plt.plot(get5y(acc22),get5(acc22), label="lr=0.0001")
    plt.plot(get5y(acc33),get5(acc33), label="lr=0.0005")
    plt.plot(get5y(acc44),get5(acc44), label="lr=0.00001")
    plt.plot(get5y(acc55),get5(acc55), label="lr=0.00005")
    plt.plot(get5y(acc66),get5(acc66), label="lr=0.000001")
    plt.plot(get5y(acc77),get5(acc77), label="lr=0.000005")
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend(loc=4)
    # plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    #plt.legend(loc=2)




def plot5_resultloss():
    plt.subplot(331)
    plt.plot(get5y(loss1),get5(loss1), label="train")
    plt.plot(get5y(loss11),get5(loss11), label="validation")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.grid()
    plt.title('lr=0.001')
    plt.subplot(332)
    plt.plot(get5y(loss2),get5(loss2), label="train")
    plt.plot(get5y(loss22),get5(loss22), label="validation")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.grid()
    plt.title('lr=0.0001')
    plt.subplot(333)
    plt.plot(get5y(loss3),get5(loss3), label="train")
    plt.plot(get5y(loss33),get5(loss33), label="validation")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.grid()
    plt.title('lr=0.0005')
    plt.subplot(334)
    plt.plot(get5y(loss4),get5(loss4), label="train")
    plt.plot(get5y(loss44),get5(loss44), label="validation")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.grid()
    plt.title('lr=0.00001')
    plt.subplot(335)
    plt.plot(get5y(loss5),get5(loss5), label="train")
    plt.plot(get5y(loss55),get5(loss55), label="validation")
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.grid()
    plt.title('lr=0.00005')
    plt.subplot(336)
    plt.plot(get5y(loss6),get5(loss6), label="train")
    plt.plot(get5y(loss66),get5(loss66), label="validation")
    plt.title('lr=0.000001')
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.grid()
    plt.subplot(337)
    plt.plot(get5y(loss7),get5(loss7), label="train")
    plt.plot(get5y(loss77),get5(loss77), label="validation")
    plt.title('lr=0.000005')
    #plt.legend(loc=2)
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.grid()
    '''
    plt.plot(get5y(),get5(loss1), label="lr=0.001")
    plt.plot(get5y(),get5(loss2), label="lr=0.0001")
    plt.plot(get5y(),get5(loss3), label="lr=0.0005")
    plt.plot(get5y(),get5(loss4), label="lr=0.00001")
    plt.plot(get5y(),get5(loss5), label="lr=0.00005")
    plt.plot(get5y(),get5(loss6), label="lr=0.000001")
    plt.plot(get5y(),get5(loss7), label="lr=0.000005")
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.grid()
    plt.legend(loc=2)
    '''
    # plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    #plt.legend(loc=2)




def readlog(path, tacc, tloss, acc, loss):
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            if 'train acc' in line:
                train_acc = float(line.split(':')[4].split(',')[0].strip())
                train_loss = float(line.split(':')[3].split(',')[0].strip())
                acc.append(train_acc)
                loss.append(train_loss)
            if 'test acc' in line:
                validation_acc = float(line.split(':')[4].split(',')[0].strip())
                validation_loss = float(line.split(':')[3].split(',')[0].strip())
                tacc.append(validation_acc)
                tloss.append(validation_loss)

            line = f.readline()


def get5(ls):
    n = -1
    lsnew = []
    for i in ls:
        n = n + 1
        if n % 15 == 0 or n == (len(ls)-1):
            lsnew.append(i)
    return lsnew

def get5y(n):
    lsnew=[]
    n=len(n)
    for i in range(n):
        if i % 15 == 0 or i == (n-1):
            lsnew.append(i)
    return lsnew

readlog(path1, acc11,loss11, acc1, loss1)
readlog(path2, acc22,loss22, acc2, loss2)
readlog(path3, acc33,loss33, acc3, loss3)
readlog(path4, acc44,loss44, acc4, loss4)
readlog(path5, acc55,loss55, acc5, loss5)
readlog(path6, acc66,loss66, acc6, loss6)
readlog(path7, acc77,loss77, acc7, loss7)
plt.figure(figsize=(10, 5))
plot5_resultacc()
plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
    wspace=None, hspace=0.9)
plt.show()

import model
import torch
import timeit
def test(x):
    start = timeit.default_timer()
    feature1 = model.get_graph_feature(x)
    end = timeit.default_timer()
    return end-start
def getx():
     x = torch.rand(2, 3, 1024).cuda()
     return x
for i in range(0,10):
    print('No '+str(i)+'Running time: %s Seconds'%(test(getx())))
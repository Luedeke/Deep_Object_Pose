#Created by: Luedeke
#Plot Graph from CSV file from DOPE
# Epoch and Loss or something lese...

import matplotlib.pyplot as plt
import csv

x = []
y = []

#path  = '/media/nils/Seagate Expansion Drive/Thesis/Trainiert/train_CandyShop2_23_05/'
path  = '/home/nils/catkin_ws/src/dope/src/training/train_FerreroKuesschen_Overfitting_1000/'
file  =  'loss_train.csv'
file2  = 'loss_test.csv'
file3  = 'test_metric.csv'

outt = 'accuracy'
with open(path + file3,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots)
    for row in plots:
	#print(row[0])
        x.append(int(row[0]))       #epoche
        y.append(float(row[2]))     #loss
        #zy.append(float(row[2]))   #batchsize

    epoche = 1
    tmp = 0
    tmp_index = 0

    tmpx = []
    tmpy = []

    #print len(x)
    for i in xrange(len(x)):
        #if(0.0001 < y[i]):
        #    print y[i]
        #print y[i], 'Epoche: ', x[i]#, ' Index: ', i
        if(x[i] == epoche):
            tmp += y[i]     #durchschnitt
            tmp_index += 1
            #print 'tmp: ', tmp, ' index: ', tmp_index
        else:
            #print 'Epoche: ', epoche
            #durchschnitt berechnen
            #print 'tmp: ', tmp, ' index: ', tmp_index
            tmpx.append(epoche)
            #if tmp != 0 and tmp_index != 0:
            tmpy.append(tmp / tmp_index)
            #print 'Loss: ', (tmp/tmp_index)
            #print 'Laenge:tmpx: ', len(tmpx)
            #print 'Laenge:tmpy: ', len(tmpy)
            #print(tmpx[epoche-1])
            #print(tmpy[epoche-1])

            tmp_index = 1
            tmp = 0
            tmp += y[i]
            epoche = x[i]
            #print 'tmp: ', tmp, ' index: ', tmp_index
        #if (i == 1650):
        #    break

    #for i in xrange(len(tmpy)):
    #   print(tmpy[i])

#print 'Laenge:tmpx: ', len(tmpx)
#print 'Laenge:tmpx: ', tmpx
#print 'Laenge:tmpy: ', len(tmpy)
#print 'Laenge:tmpy: ', tmpy
plt.plot(tmpx,tmpy, label='Overfitting Metric')
plt.xlabel('epoche')
plt.ylabel(outt)
plt.title('DOPE')
plt.legend()
plt.show()

#Created by: Luedeke
#Plot Graph from CSV file from DOPE
# Epoch and Loss or something lese...

import matplotlib.pyplot as plt
import csv


path  = '/media/nils/Seagate Expansion Drive/Thesis/Trainiert/train_CandyShop_14_05_ganz_neu/'
#file  =  path + 'loss_train.csv'
#file2  = path + 'loss_test.csv'
file  =  'loss_train.csv'
file2  = 'loss_test.csv'


files = [file, file2]

for f in files:
    x = []
    y = []

    tmpx = []
    tmpy = []
    with open(path + f,'r') as csvfile:
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
                tmpx.append(epoche)
                tmpy.append(tmp / tmp_index)

                #print 'tmp: ', tmp, ' index: ', tmp_index
                #print 'Loss: ', (tmp/tmp_index)
                #print 'Laenge: ', len(tmpx)
                #print(tmpx[epoche-1])
                #print(tmpy[epoche-1])

                tmp_index = 1
                tmp = 0
                tmp += y[i]
                epoche = x[i]
                #print 'tmp: ', tmp, ' index: ', tmp_index

            #if (i == 1650):
            #    break

    plt.plot(tmpx,tmpy, label=f)
        #for i in xrange(len(tmpy)):
        #   print(tmpy[i])



plt.xlabel('epoche')
plt.ylabel('loss')
plt.title('DOPE')
plt.legend()
plt.show()

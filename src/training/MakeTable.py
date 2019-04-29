import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('loss_train.csv','r') as csvfile:
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

    #print 0.000099009012047 #9.9009012047e-05
    #print 0.000153484608745

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

    #for i in xrange(len(tmpy)):
    #   print(tmpy[i])


plt.plot(tmpx,tmpy, label='DOPE: Fine-Tuning Layer1')
plt.xlabel('epoche')
plt.ylabel('loss')
plt.title('DOPE')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import csv

#data = timestamp \t cureentOption \t fps \t expOUtputRate \t delay \t time averg performance \n






result1 = [[] for i in range(8)]
with open('RESULT','r') as csvfile:
    plots = csv.reader(csvfile, delimiter='\t')
    for row in plots:
        for i in range(len(row)): result1[i].append(float(row[i]) if i != 1 else row[i] )

result2 = [[] for i in range(8)]
with open('RESULT2','r') as csvfile:
    plots = csv.reader(csvfile, delimiter='\t')
    for row in plots:
        for i in range(len(row)): result2[i].append(float(row[i]) if i != 1 else row[i] )



result_static_opt = [[] for i in range(8)]
with open('RESULT_static_opt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter='\t')
    for row in plots:
        for i in range(len(row)): result_static_opt[i].append(float(row[i]) if i != 1 else row[i] )


result_static_slow = [[] for i in range(8)]
with open('RESULT_static_slow', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter='\t')
    for row in plots:
        for i in range(len(row)): result_static_slow[i].append(float(row[i]) if i != 1 else row[i])

result_static_fast = [[] for i in range(8)]
with open('RESULT_static_fast','r') as csvfile:
    plots = csv.reader(csvfile, delimiter='\t')
    for row in plots:
        for i in range(len(row)): result_static_fast[i].append(float(row[i]) if i != 1 else row[i] )





plt.hold(True)
screenw = 500
screenh = 400
startposx = 70
startposy = 70
interval = 20
intervalh = 40
#====================================================================================================================
fig1 = plt.figure(1);
plt.get_current_fig_manager().window.setGeometry(startposx,startposy,screenw,screenh)
plt.plot(result1[0][:],result1[1][:] , label = "dynamic (time constraint 0.5sec)")
plt.plot(result2[0][:],result2[1][:] , label = "dynamic (time constraint 0.3sec)")
plt.plot(result_static_slow[0][:],result_static_slow[1][:] , label = "static opt(model2)")
plt.plot(result_static_fast[0][:],result_static_fast[1][:] , label = "static fast(model5)")
#plt.ylim([0, 1])
plt.xlim([0,140])
plt.ylabel('Model')
plt.xlabel('Time (sec)')
plt.legend()
#plt.grid(True)

#====================================================================================================================




#====================================================================================================================
fig2 = plt.figure(2);
plt.plot(result1[0][:],result1[3][:] , linewidth = 0.2,label = "1's expected output rate")
plt.plot(result1[0][:],result1[2][:] , linewidth= 0.2, label = "1's output rate")


plt.plot(result2[0][:],result2[3][:] , linewidth = 0.2,label = "2's expected output rate")
plt.plot(result2[0][:],result2[2][:] , linewidth= 0.2, label = "2's output rate2")
plt.ylim([0, 20])
plt.ylabel('output rate(fps)')
plt.xlabel('Time (sec)')
plt.legend()
#plt.grid(True)
#====================================================================================================================






#====================================================================================================================
fig2 = plt.figure(3);
plt.get_current_fig_manager().window.setGeometry(startposx+(screenw+interval)*0,startposy+(screenh+intervalh)*1,screenw,screenh)




plt.plot(result1[0][:],result1[6][:],color = 'black',linewidth = 1, linestyle = '--',label = "time constraint" )
plt.plot(result2[0][:],result2[6][:],color = 'black',linewidth = 1, linestyle = '--' )

plt.plot(result1[0][:],result1[4][:] ,linewidth = 0.5,label ="dynamic (time constraint 0.5sec)")
plt.plot(result2[0][:],result2[4][:] ,linewidth = 0.5, label = "dynamic (time constraint 0.3sec)")


plt.plot(result_static_opt[0][:],result_static_opt[4][:],linewidth = 0.5, linestyle = '-',label = "static opt(model4)" )
plt.plot(result_static_fast[0][:],result_static_fast[4][:],linewidth = 0.5, linestyle = '-',label = "static fast(model5)" )
plt.plot(result_static_slow[0][:],result_static_slow[4][:],linewidth = 1, linestyle = '-',label = "static slow(model3)" )



plt.ylim([0, 0.8])
plt.xlim([0,120])
plt.ylabel('Delay (sec)')
plt.xlabel('Time (sec)')
plt.legend()
plt.legend(loc='upper right')
#plt.grid(True)
#====================================================================================================================





#====================================================================================================================
fig2 = plt.figure(4);
plt.legend(loc='upper right')
plt.get_current_fig_manager().window.setGeometry(startposx+(screenw+interval)*1,startposy+(screenh+intervalh)*1,screenw,screenh)
plt.plot(result1[0][:],result1[5][:] , label = "dynamic (time constraint 0.5sec)")
plt.plot(result2[0][:],result2[5][:] , label = "dynamic (time constraint 0.3sec)")


plt.plot(result_static_opt[0][:],result_static_opt[5][:] , label = "static opt(model4)")
plt.plot(result_static_fast[0][:],result_static_fast[5][:] , label = "static fast(model5)")
plt.plot(result_static_slow[0][:],result_static_slow[5][:] , label = "static slow(model3)")


#plt.ylim([0, 1])
plt.xlim([0,120])
plt.ylabel('Time Average Performance')
plt.xlabel('Time (sec)')
plt.legend()
#plt.grid(True)
#====================================================================================================================






#====================================================================================================================
fig2 = plt.figure(5);
plt.get_current_fig_manager().window.setGeometry(startposx+(screenw+interval)*2,startposy+(screenh+intervalh)*0,screenw,screenh)


heavystress = [ [] for i in range(2)]
heavystress[0] = [i for i in range(180)]
heavystress[1] = [0 for i in range(180)]
heavystress[1][80:100] = [20 for i in range(20)]


extremeheavystress = [ [] for i in range(2)]
extremeheavystress[0] = [i for i in range(180)]
extremeheavystress[1] = [0 for i in range(180)]
extremeheavystress[1][120:140] = [20 for i in range(20)]

width = 1
plt.bar(heavystress[0][:],heavystress[1][:], width, color = "red", label = "GPU stress")
#plt.bar(extremeheavystress[0][:],extremeheavystress[1][:], width, color = "red", label = "extremely stressed")


plt.plot(result1[0][:],result1[7][:] , label = "input rate")



plt.ylim([0, 20])
plt.xlim([0,120])
plt.ylabel('Time Average Performance')
plt.xlabel('Time (sec)')
plt.legend()
#plt.grid(True)
#====================================================================================================================


#plt.legend("Static Low Sampling Rate", "Static High Sampling Rate","Dynamic adaptation","Overflow", 'Location','NorthWest')
plt.show()

import numpy as np
import random
#####################################################################
def Score_prediction_specific_TR(prediction, label_assigned, y):
    answer = []
    for i in range(len(prediction)):
        if prediction[i]>0:
            answer.append(1)
        else:
            answer.append(-1)
    #Scoring
    score = 0
    for i in range(len(prediction)):
        if answer[i]==1 and y[i] == label_assigned:
            score+=1
        if answer[i]==-1 and y[i]!= label_assigned:
            score+=1
    return score/len(prediction)

#####################################################################
def Assign_answer_list_of_ALL_TR(prediction,answer_list):
    for i in range(len(prediction)):
        if prediction[i]>0:
            answer_list[i]=1
            
    return answer_list

def Answer_defined(answer_list):
    DE = 0
    for i in range(len(answer_list)):
        if answer_list[i]>0:
            DE+=1
    
    return DE/len(answer_list), DE

#####################################################################

#####################################################################

def Score(prediction,assigned_label,y):
    S = 0
    for i in range(len(prediction)):
        #print(y[i])
        if prediction[i]>=0 and int(y[i])==int(assigned_label):
            S+=1
        
        if prediction[i]<=-0 and int(y[i])!=int(assigned_label):
            S+=1
    return S/len(prediction)

def Get_Sample(prediction,x,y):
    waiting_list = []
    for i in range(len(prediction)):
        if prediction[i]>0:
            #print("!")
            waiting_list.append(i)
    
    #To see if labelling is correct
    zeg = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(waiting_list)):
        zeg[int(y[waiting_list[i]])]+=1
    index = 0
    maximum = 0
    if len(waiting_list)==0:
        print("ERROR IN GET SAMPLE", prediction)
        return -1
    kl = int(random.randint(0,len(waiting_list)-1))
    for i in range(len(zeg)):
        if zeg[i]>maximum:
            maximum = zeg[i]
            index = i
    print(zeg,"\n")
    print("Real_label is: ",index,"\nCurrent label is: ",waiting_list[kl],"|",y[waiting_list[kl]])
            
    return waiting_list[kl]
        
        
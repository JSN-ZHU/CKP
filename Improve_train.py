import numpy as np
import Network as net
import Score_Metrics
import random

#####################################################################
def Single_Improve_train(x,translator,translators,epochs,batch_size):####
    
    for j in range(len(translators)):
        prediction = translator.predict(x)
        temp_x = []
        temp_y = []  #Self labelling
        balance = 0
        prediction_of_trained = translators[j].predict(x)
        for i in range(batch_size):
            if len(temp_x)<=batch_size:
                for z in range(5):
                    index = 0
                    k = random.randint(0,len(prediction)-1)
                      #Use balanced class to solve the problem of unbalanced possible self-labelling.
                    if (prediction[k]>index and prediction_of_trained[k]<0) and balance>=0:  #Possible repetition of same data but unknown result
                        temp_x.append(x[i])
                        temp_y.append(1)
                        balance-=2
                        #print("TRUE")

                    if (prediction[k]<-index and prediction_of_trained[k]>0) and balance<=0:
                        temp_x.append(x[i])
                        temp_y.append(-1)
                        balance+=1
                
    temp_y = np.asarray(temp_y).reshape(len(temp_y),1)
    translator.fit(temp_x,temp_y,epochs)
    print("\n|||||||||||",len(temp_x), batch_size,balance)
    if len(temp_x)<batch_size*0.01:  #
        #Single_Improve_train(x,translator,translators,epochs,batch_size)
        pass
    return

    
def Improve_train(x,translator,end_epochs,C_TR):
    start_epochs = int(end_epochs*0.1)+1  # With multiple times of training, the over-fitting can be reduced.
    for i in range(4):  #0~8
        temp_epochs = int(i*end_epochs*0.2)+start_epochs
        Single_Improve_train(x,translator,C_TR,temp_epochs,(i+2)*600)
        if i==0:
            print("")
            
            Single_Improve_train(x,translator,C_TR,50,20)
    
    temp_epochs = end_epochs
    print("")
    #Single_Improve_train(x,translator,C_TR,temp_epochs,len(x)) #Train more and more to simulate the process of K-Means
    return

#####################################################################

def Combine(translators,mem):
    for i in range(len(mem)-1):
        translators.remove(translators[mem[-1]])
    return translators


def Combine_EX_TR(x,translators):
    answers = []
    for i in range(len(translators)):
        prediction = translators[i].predict(x)
        
        for j in range(len(prediction)):
            if prediction[j]>0:
                prediction[j]=2
                
            if prediction[j]<0:
                prediction[j]=-2
                
        answers.append(prediction)
    
    for i in range(len(answers)):
        mem = []
        mem.append(i)
        for j in range(i,len(answers)):
            s = 0
            for k in range(len(answers[i])):
                if answers[i][k]==answers[j][k]:
                    s+=1
            if s/len(answers[i])>0.8:
                mem.append(j)
        print("Combined", mem)
        translators = Combine(translators,mem)
    
    return translators

#####################################################################

#Use of self labelling is a kind of using 'specialities'. But in this task, it will be possible to eliminate the disadvantages of this 'specialities'
#It can be possible for this method to recognizes all data as 'negative', which means a 80% accurace around. But this is not a 'real' accuracy. The way to solve this in function start at line 6
import mnist_loader
import Network as net
import Score_Metrics
import Improve_train
import random
import numpy as np
import os

#To notify here, this method can eventually turn all the supervised ML to unsupervised
Core = net.Sequential()  #Better if use a pretrained network to narrow down the possible targeting class (e.g. not targeting color but the shape)
#Will 'pretrain' in following processes
#Also, the Core is constructed with only ANN and RELU, 
# so please do not blame the low accuracy of this model(YOU CAN REPLACE WITH OTHER MODELS WITH THIS)
#And this is a method based module, so currently I am not planning to improve its accuracy
#####################################################################

#Get data 
def dataloader(dataset_name = "mnist"):
    if dataset_name == "mnist":
        x,y = mnist_loader.load_mnist('C:/Users/DELL/Desktop/All_Files/Programs/MNIST')
    return x,y
    pass

#####################################################################

def Get_Certain_label(x,y,label,data_n):
    answer_x = []
    answer_y = []
    counter = 0
    balance_counter = -1
    for i in range(len(x)):
        if len(answer_x)<data_n:
            if balance_counter<=0:
                if y[i]==int(label):
                    answer_x.append(x[i])
                    answer_y.append(1)
                    balance_counter+=1
        
            if balance_counter>0:
                if y[i]!=int(label):
                    answer_x.append(x[i])
                    answer_y.append(-1)
                    balance_counter-=1
                
        #Balanced training set, to optimize the accuracy
    #print(balance_counter,"||","See balanced or not")
    answer_y = np.asarray(answer_y).reshape(len(answer_y),1) #Standarize
    print(len(answer_x),"!!!!!!!",len(x))
    return answer_x,answer_y





#To pretrain
def Core_train(in_nodes, hid_nodes ,out_nodes ,x,y,epochs):
    Core = net.Sequential()
    Core.add(net.DenseLayer(in_nodes,hid_nodes,lr = 0.2))
    Core.add(net.ActivationLayer(net.Sigmoid))
    Core.add(net.DenseLayer(hid_nodes,hid_nodes*2,lr = 0.2))
    Core.add(net.ActivationLayer(net.Sigmoid))
    #Core.add(net.DenseLayer(hid_nodes*2,hid_nodes*2,lr = 0.2))
    #Core.add(net.ActivationLayer(net.Sigmoid))
    Core.add(net.DenseLayer(hid_nodes*2,hid_nodes,lr = 0.2))
    Core.add(net.ActivationLayer(net.Sigmoid))
#Below is the same structure/ similar structure an ocsvm used
    Core.add(net.DenseLayer(hid_nodes,out_nodes*10,lr=0.3))
    Core.add(net.ActivationLayer(net.Sigmoid))
    Core.add(net.DenseLayer(out_nodes*10,out_nodes,lr=0.2))

#To 'pretrain', the specific random data class is extracted for training
    for kll in range(10):  #Maximum 20 in this dataset, in fact can set as len(x)
        
        z = 300
        temp_x = []
        temp_y = []
        label_chosen = y[0]#input("Input the label you want to choose to initiate: ")
        #print("Label chosen: ",label_chosen)
        temp_x , temp_y = Get_Certain_label(x,y,label_chosen,int(z*(kll+1)))
        #################
        Core.fit(temp_x,temp_y,epochs)
        prediction = Core.predict(x)
        #print('',y[0])
        Current_Score = Score_Metrics.Score(prediction,label_chosen,y)
        print("")
        print("The following is current score")
        print(Current_Score)
        print("_____________________________________________")
        pass
    return Core
    pass





#####################################################################

#Building CKP
def build_TR(in_nodes):
    
    pass


class CKP_M():
    def __init__(self):
        self.CKP_Translators = []
        self.CKP_labels = []
        
    def add_T(self,translator,index_P):
        self.CKP_Translators.append(translator)
        self.CKP_labels.append(index_P)   #Can use dictionary here, but this one is easier to understand
        pass
    


epochs = 700
x,y = dataloader()
hid_nodes = 200
out_nodes = 1
Core = Core_train(784,hid_nodes,1,x,y,epochs)
T_Layer_N = 3
Translator = CKP_M()
#Transfer of pretrained information into designed translator
temp = net.Sequential()
for i in range(T_Layer_N): #Get 3 last layers, expected to be orig translator
    temp.add(Core.remove(int(Core.layersLevel)-T_Layer_N+i))
    
    
    
processed_x = Core.predict(x)
label = Score_Metrics.Get_Sample(temp.predict(processed_x),x,y)
print(Score_Metrics.Answer_defined(temp.predict(processed_x)))
Translator.add_T(temp,label)
#Next will be further training and building of other translators
Translator_num = int(input("Enter the translator number you want: "))-1

def Single_OC_SVM_Train(real_x,y,x,answer_list,CK_T):
    temp = net.Sequential()
    temp.add(net.DenseLayer(hid_nodes,out_nodes*10,lr=0.3))
    temp.add(net.ActivationLayer(net.Sigmoid))
    temp.add(net.DenseLayer(out_nodes*10,out_nodes,lr=0.2))
    temp_x = []
    temp_y = []
    
    while True:
        z = random.randint(0,len(answer_list)-1)
        if answer_list[z]<=0:
            temp_x.append(x[z])
            temp_y.append(1)
            break
    while True:                                 #data extraction, for partial training
        z = random.randint(0,len(answer_list)-1)
        if answer_list[z]>0:
            temp_x.append(x[z])
            temp_y.append(-1)
        
            break
    temp_y = np.asarray(temp_y).reshape(len(temp_y),1)
    temp.fit(temp_x,temp_y,50)
    print("")
    Improve_train.Improve_train(x,temp,100,CK_T)
    prediction = temp.predict(x)
    print(len(prediction))
    #Following is used to supervise the unsupervised network --> how it is doing now --> and have no interference with the code
    if len(prediction)>0:
        current_label = Score_Metrics.Get_Sample(prediction,real_x,y)
        Current_Score = Score_Metrics.Score(prediction,current_label,y)
        print("")
        print("Current defined answer",Score_Metrics.Answer_defined(prediction))
        print("")
        '''
        print("The following is current score after pretrained")  #It seems that it has some problem here.
        print(Current_Score, len(y))
        '''
        print("_____________________________________________")
        #print(prediction)
        return temp
    else:
        print("Error: ",temp.predict(x))
        

#next is the building of CKP, I name it as translator according to the defination of a single functional unit as input, process translate
while len(Translator.CKP_Translators)<Translator_num:
    print("\nNow is ",len(Translator.CKP_Translators)," Translator !!")
    processed_x = Core.predict(x)
    C_TR = Translator.CKP_Translators
    answer_list = C_TR[0].predict(processed_x)
    answer_list = Score_Metrics.Assign_answer_list_of_ALL_TR(answer_list,answer_list)
    
    for j in range(len(C_TR)):
        prediction = C_TR[j].predict(processed_x)
        answer_list = Score_Metrics.Assign_answer_list_of_ALL_TR(prediction,answer_list)
    
    temp_TR = Single_OC_SVM_Train(x,y,processed_x,answer_list,C_TR)
    temp_TR_prediction = temp_TR.predict(processed_x)
    defined, __1 = Score_Metrics.Answer_defined(temp_TR_prediction)
    
    if defined>0.8 or defined<0.05:
        temp_TR = None
        print("Breaken point while found quantity is: ", __1)
    print("\nCurrently TR defined: ",defined)
    
    if temp_TR != None:
        answer_list = []
        for z in range(len(processed_x)):
            answer_list.append(-1)
        for z in range(len(C_TR)):
            prediction = C_TR[z].predict(processed_x)
            answer_list = Score_Metrics.Assign_answer_list_of_ALL_TR(prediction,answer_list)
        
        Total_defined = Score_Metrics.Answer_defined(answer_list)
        Total_defined = Score_Metrics.Answer_defined(temp_TR_prediction)
        print("\nTotal defined: ",Total_defined)
        label = Score_Metrics.Get_Sample(temp_TR_prediction,x,y)
        Translator.add_T(temp_TR,label)
    #Translator.CKP_Translators = Improve_train.Combine_EX_TR(processed_x,Translator.CKP_Translators)
    
real_labels = []###
#Get all real labels of the Translators
for i in range(len(C_TR)):
    prediction = C_TR[i].predict(processed_x)
    zeg = [0,0,0,0,0,0,0,0,0,0]
    counter = 0
    for j in range(len(prediction)):
        if prediction[j]>0:
            zeg[y[j]]+=1
            counter+=1
    print("Current translator in verification is : ", i+1)
    max_c = 0
    index = 0
    for j in range(len(zeg)):
        if zeg[j]>max_c:
            index = j
            max_c = zeg[j]
            
    real_labels.append(index)
    print("In field score is: ",max_c/counter)

all_prediction = C_TR[0].predict(processed_x) #To get the final score
all_Answers = all_prediction   #Initialization full rankage
for i in range(len(C_TR)):
    current_prediction = C_TR[i].predict(processed_x)
    for z in range(len(current_prediction)):
        if current_prediction[z]>=all_prediction[z]:
            all_Answers[z]=real_labels[i]
            all_prediction[z]=current_prediction[z]

#Here try to combine the extra translators
Improve_train.Combine_EX_TR(processed_x,C_TR)

Final_Score = 0
for i in range(len(y)):
    if int(y[i])==int(all_Answers[i]):
        Final_Score+=1

'''
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("Final accuracy is: ",Final_Score/len(y))     #Accuracy of final score is still low.
'''

Improve_train.Combine_EX_TR(processed_x,C_TR)
#At the end of the code, I want to notify one interesting thing.
#This code is recognizing ABSTRACT PATTERN! This means that it can distinguish a dataset with basic shapes without labelling(If you start with a pretrained model and go directly to build translators, but its unlucky that I don't have a dataset to verify this.)

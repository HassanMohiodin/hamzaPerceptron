import pandas as pd
import numpy as np
import random 
import sys 

i=0
sum=0
counter=0
threshold = 0.2
x0=-1
total_error = 0
weights = []
rate_of_learning = 0.1
arg = "train"#sys.argv[1]
file_name = "trainFile"#sys.argv[2]
no_of_epochs = 5000 #sys.argv[3] 
Accuracy = 0
Precision = 0
recall = 0




def train():
    
    global weights
    
    final_data = data_processed(file_name)
    weights = [random.uniform(-0.5,0.5) for i in range(0,len(final_data.columns))]
    main_fun(final_data)
    print_cm(final_data,weights)

def test():
    
    global weights
    count = 0
    
    final_data = data_processed(file_name)
    for appending in range(len(final_data.columns)):
        weights.append(0)
        
    file = open("Project7406.txt", "r")
    for line in file:
            weights [count] = float(line)
            count = count + 1
    print_cm(final_data,weights)
    save_to_file()
    

def cmd_arg(check):
    if(check == "test"):
        test()
    else:
        train()


def data_processed(f_name):
    global weights
    data = pd.read_csv("C:/Users/hamza/Desktop/New folder/"+ f_name +".csv",header=None)
      
    length = len(data.columns) 
    temp_array = []
    temp = 0 
    
    
    for i in range(length):
        temp_array.append(0)
        
    
    while temp < length:
        column = data[temp].values
        max_value = np.max(column)
        temp_array[temp] = column/max_value
        temp = temp + 1
    
        
    df = pd.DataFrame(temp_array)
    df.to_csv("C:/Users/hamza/Desktop/New folder/changed.csv" ,  header=False, index=False)
    
    changed_data = pd.read_csv("C:/Users/hamza/Desktop/New folder/changed.csv",header=None)
    df1 = pd.DataFrame(changed_data)
    data_used = df1.transpose()
    return data_used

def threshold_fn(param_sum):  
    threshold_value = param_sum - threshold * x0
    if threshold_value < 0:
        return 0
    else:
        return 1

def calculating_error(param_predicted):
    global value
    
    return value - param_predicted


def predict(weights,X):
    predictions = []
    counter = 0
    sum = 0
    
    while counter < X.shape[0]:
        new_one = np.append(X.iloc[counter,0:X.shape[1]-1].values,x0)
        i = 0
        sum = 0
        for each_data in new_one :
            sum = sum + (each_data * weights[i])
            i=i+1
        
        predicted_value = threshold_fn(sum)
        predictions.append(predicted_value)
        counter= counter+1
    return predictions


def confusion_matrix(outputs,predictions):
    cm = [[0,0],[0,0]]
    
    for i in range(0,len(predictions)):
        cm[int(predictions[i])][int(outputs[i])] += 1
    return cm


def main_fun(final_data):
    
    f = open('output_data.txt', 'r+')
    f.truncate(0)
    
    file_output = open ("output_data.txt","a")
    
    global counter
    global sum
    global total_error
    global i
    global weights
    global value
    global no_of_epochs
    
    for epoch in range(int(no_of_epochs)): 
    #    print("predicted Values")
        while counter < final_data.shape[0]:
            new_one = np.append(final_data.iloc[counter,0:final_data.shape[1]-1].values,x0)
            for each_data in new_one :
                sum = sum + (each_data * weights[i])
                i=i+1
            predicted_value = threshold_fn(sum)
            
        
            value =  final_data[final_data.shape[1]-1][counter]
            error = calculating_error( predicted_value )
            total_error = total_error + abs(error)
            i=0 
            
            for each_data in new_one:
                weights[i] = weights [i] + rate_of_learning * error * each_data
                i=i+1
            i=0
            sum =0 
            counter= counter+1
            
            write_to_file(epoch,new_one,predicted_value, value ,error,file_output)
    
    #    print("error=",total_error)
        total_error=0
        counter = 0
        file = open ("Project7406.txt","w")
        for weight in weights:
            file.write(str(weight) + "\n")
        file.close()
    file_output.close()
    
    
    
def print_cm(final_data,weights):
    
    global Accuracy
    global Precision
    global recall
    
    predictions = predict(weights,final_data)
    #print('predictions',predictions)
    cm = confusion_matrix(final_data.iloc[:,-1].values,predictions)
    Accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100
    Precision = (cm[0][0])/(cm[0][0]+cm[0][1])*100
    recall = (cm[0][0])/(cm[0][0]+cm[1][0])*100
    print("Accuracy=",Accuracy,"%")
    print("Precision=",Precision,"%")
    print("recall=",recall,"%")
    #print("Weights=",weights)
    
def save_to_file():
    
    f = open('Project.txt', 'r+')
    f.truncate(0)
    
    global Accuracy
    global Precision
    global recall
    
    file = open ("Project.txt","w")
    file.write("Accuracy="  + str(Accuracy) + " \n"  + "Precision=" + str(Precision) + " \n"  + "Recall=" + str(recall))
    file.close()
    
    
def print_data(file , inputs):
    file.write(str(inputs) + "  ")
    
def write_to_file(epoch,new_one,predicted_value, value ,error,file):
    
    
    file.write(str(epoch) + "            ")
    for each_data in new_one:
         con_dec = float("{0:.2f}".format(each_data))
         print_data(file , con_dec)
    file.write("            ")
    for each_weight in weights:
        con_dec = float("{0:.2f}".format(each_weight))
        print_data(file , con_dec)
    file.write("            " + str(value) )
    file.write("            " + str(predicted_value) )
    file.write("            " + str(error) )
    file.write("\n")


cmd_arg(arg)


    
    
        
    
 

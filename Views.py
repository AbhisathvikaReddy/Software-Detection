from django.shortcuts import render
import numpy as np
# Create your views here.
import pickle

def home(request):
    if request.method == "POST":
        input1 = request.POST["input1"]
        input2 = request.POST["input2"]
        input3 = request.POST["input3"]
        input4 = request.POST["input4"]
        input5 = request.POST["input5"]
        input6 = request.POST["input6"]
        input7 = request.POST["input7"]
        input8 = request.POST["input8"]
        input9 = request.POST["input9"]
        input10 = request.POST["input10"]
        input11 = request.POST["input11"]
        input12 = request.POST["input12"]
        input_features=[float(input1),float(input2),float(input3),float(input4),float(input5),float(input6),float(input7),float(input8),float(input9),float(input10),float(input11),float(input12)]
        input_features=np.array(input_features).reshape(1,-1)
        with open(r'"C:\Users\10nik\Downloads\SOFTWARE DETECTION\majorproject.py"\model1.pkl' , 'rb') as file:
            rf = pickle.load(file) 
        with open(r'"C:\Users\10nik\Downloads\SOFTWARE DETECTION\majorproject.py"\model2.pkl' , 'rb') as file:
            lr = pickle.load(file) 
        with open(r'"C:\Users\10nik\Downloads\SOFTWARE DETECTION\majorproject.py"\model3.pkl' , 'rb') as file:
            ensemble_model = pickle.load(file)
        y_pred= rf.predict(input_features)
        y_pred2= lr.predict(input_features)
        input_linear=[y_pred[0],y_pred2[0]]
        input_linear=np.array(input_linear).reshape(1,-1)
        y_final = ensemble_model.predict(input_linear)
        result = int(y_final.round())
        print(y_final) 
        #result = ensemble_model.predict([[input1,input2,input3,input4,input5,input6,input7,input8,input9,input10,input11,input12,input13,input14,]])
        return render(request,'home.html',{'result':result})
    return render(request,'home.html')
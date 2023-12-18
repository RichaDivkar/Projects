#To Build Machinen Learning Application to Predict the Loan Status of Application.

import streamlit as st
import pandas as pd
import numpy as np
import base64
import pickle  #to load a saved modelimport base64  #to open .gif files in streamlit app
import joblib

@st.cache_resource  
def get_fvalue(val):    
    feature_dict = {"No":0,"Yes":1}   
    for key,value in feature_dict.items():        
        if val == key:            
            return value
def get_value(val,my_dict):    
    for key,value in my_dict.items():        
        if val == key:            
            return value

app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) #two pages

if app_mode=='Home':    
    st.title('LOAN PREDICTION :')      
    st.image('Loan_status_prediction/img1.jpg')  
    st.markdown(
    '<p style="background-color:cornsilk;padding:10px">'
    'I created this machine learning web application for loan prediction based on a loan dataset shown below. '
    'Its primary purpose is to forecast or estimate the outcome of a loan application. '
    'This application relies on the information within the loan_dataset to provide predictions regarding whether a loan is likely to be approved or not. '
    'The goal behind this development is to assist lenders in making more informed decisions by using historical data to predict potential loan statuses accurately.'
    '</p>',
    unsafe_allow_html=True
) 
    st.subheader('Dataset :')    
    data=pd.read_csv('Loan_status_prediction/loan_data_set.csv')    
    st.write(data.head())    
    st.markdown('Applicant Income versus Loan Amount for the First 10 Applicants')    
    st.bar_chart(data[['ApplicantIncome', 'LoanAmount']].head(10), color=['#DDA0DD', '#8B008B'])
    st.write("To access detailed information about the model, please check the Git repository.")
    st.markdown("[Git Repository Link](https://github.com/RichaDivkar/Projects.git)")
    st.subheader('To check your loan status, please navigate to the prediction section in the sidebar.')

elif app_mode == 'Prediction':    
    st.image('Loan_status_prediction/Loan_pred.jpg')    
    st.subheader("Sir/Ma'am, please ensure all necessary information is filled out completely to receive a response to your loan request.") 
    st.markdown("Please click the 'Predict' button after entering the information.")
    st.sidebar.header("Informations about the client :")   

#For Feature Engineering
    gender_dict = {"Male":1,"Female":0}    
    feature_dict = {"No":0,"Yes":1}    
    edu={'Graduate':0,'Not Graduate':1}    
    prop={'Rural':1,'Urban':2,'Semiurban':3} 

#input values
    ApplicantIncome=st.sidebar.slider('ApplicantIncome',0,100000,0,)    
    CoapplicantIncome=st.sidebar.slider('CoapplicantIncome',0,100000,0,)    
    LoanAmount=st.sidebar.slider('LoanAmount in K$',9.0,700.0,200.0)    
    Loan_Amount_Term=st.sidebar.selectbox('Loan_Amount_Term',(12.0,36.0,60.0,84.0,120.0,180.0,240.0,300.0,360.0,420.0,480.0,540.0))    
    Credit_History=st.sidebar.radio('Credit_History',(0.0,1.0))    
    Gender=st.sidebar.radio('Gender',tuple(gender_dict.keys()))    
    Married=st.sidebar.radio('Married',tuple(feature_dict.keys()))    
    Self_Employed=st.sidebar.radio('Self Employed',tuple(feature_dict.keys()))    
    Dependents=st.sidebar.radio('Dependents',options=['0','1' , '2' , '3+'])    
    Education=st.sidebar.radio('Education',tuple(edu.keys()))    
    Property_Area=st.sidebar.radio('Property_Area',tuple(prop.keys()))

#one-hot encoding
    class_0 , class_3 , class_1,class_2 = 0,0,0,0    
    if Dependents == '0':        
        class_0 = 1    
    elif Dependents == '1':        
        class_1 = 1    
    elif Dependents == '2' :        
        class_2 = 1    
    else:        
        class_3= 1  

    Rural,Urban,Semiurban=0,0,0    
    if Property_Area == 'Urban' :        
        Urban = 1    
    elif Property_Area == 'Semiurban' :        
        Semiurban = 1    
    else :        
        Rural=1


    ApplicantIncome_ans=(ApplicantIncome-0)/(100000-0)
    CoapplicantIncome_ans=(CoapplicantIncome-0)/(100000-0)
    LoanAmount_ans=(LoanAmount-9)/(700-9)
    Loan_Amount_Term_ans=(Loan_Amount_Term-12)/(540-12)


    data1={    
        'Gender':Gender,    
        'Married':Married,    
        'Dependents':[class_0,class_1,class_2,class_3],    
        'Education':Education,    
        'ApplicantIncome':ApplicantIncome_ans,    
        'CoapplicantIncome':CoapplicantIncome_ans,    
        'Self Employed':Self_Employed,    
        'LoanAmount':LoanAmount_ans,    
        'Loan_Amount_Term':Loan_Amount_Term_ans,    
        'Credit_History':Credit_History,    
        'Property_Area':[Rural,Urban,Semiurban],    
        }    
    feature_list=[ApplicantIncome_ans,CoapplicantIncome_ans,LoanAmount_ans,Loan_Amount_Term_ans,Credit_History,
                get_value(Gender,gender_dict),
                get_fvalue(Married),
                data1['Dependents'][0],
                data1['Dependents'][1],
                data1['Dependents'][2],
                data1['Dependents'][3],
                get_value(Education,edu),
                get_fvalue(Self_Employed),
                data1['Property_Area'][0],
                data1['Property_Area'][1],
                data1['Property_Area'][2]]    

    single_sample = np.array(feature_list).reshape(1,-1)

    if st.button("Predict"):        
        file_ = open("Loan_status_prediction/6m-rain.gif", "rb")        
        contents = file_.read()        
        data_url = base64.b64encode(contents).decode("utf-8")        
        file_.close()        
        file = open("Loan_status_prediction/noloans.gif", "rb")        
        contents = file.read()        
        data_url_no = base64.b64encode(contents).decode("utf-8")        
        file.close()        
        # loaded_model = pickle.load(open('random_forest_model.sav', 'rb'))      
        # random_forest_model = joblib.load('random_forest_model.sav')
        loaded_model = load('Project1_model.joblib')
        # with open('Loan_status_prediction/Project1_model.pkl', 'rb') as file:
        #     loaded_model = pickle.load(file)
        
        prediction = loaded_model.predict(single_sample)
        #st.write(prediction)

        if prediction[0] == 0 :            
            st.error(    'According to our Calculations, you will not get the loan from Bank'    )            
            st.markdown(
                    f'<img src="data:image/gif;base64,{data_url_no}" alt="cat gif" width="400">',
                    unsafe_allow_html=True,
                )        
        elif prediction[0] == 1 :
            st.balloons()            
            st.success(    'Congratulations!! you will get the loan from Bank'    )            
            st.markdown(    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width="400">',    
                        unsafe_allow_html=True,    )

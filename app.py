ofrom tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


global filename
global df, X_train, X_test, y_train, y_test
global ada_acc, rf_acc, knn_acc, gnb_acc, lr_acc, svm_acc, lda_acc, smote_enn_acc

def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    
    # Replace '?' with NaN
    df.replace(0, np.nan, inplace=True)

    # Fill missing values with mode for each column
    df.fillna(df.mode().iloc[0], inplace=True)
    
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(len(df)) + "\n")

def splitdataset(): 
    global df, X_train, X_test, y_train, y_test

    # Encode string columns to numerical values
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])

    X = np.array(df.drop(["Class/ASD Traits "], axis=1))
    y = np.array(df["Class/ASD Traits "])
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    
    # Display dataset split information
    text.delete('1.0', END)
    text.insert(END, "Dataset split\n")
    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(X_train)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(X_test)) + "\n")
    
    # Display shapes of X_train, X_test, y_train, y_test
    text.insert(END, "\nShape of X_train: " + str(X_train.shape) + "\n")
    text.insert(END, "Shape of X_test: " + str(X_test.shape) + "\n")
    text.insert(END, "Shape of y_train: " + str(y_train.shape) + "\n")
    text.insert(END, "Shape of y_test: " + str(y_test.shape) + "\n\n")

def adaboost():
    global ada_acc
    ada = AdaBoostClassifier(n_estimators=100, random_state=0)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    ada_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for AdaBoost is {ada_acc * 100}%\n'
    text.insert(END, result_text)

def random_forest():
    global rf_acc,rf
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for Random Forest is {rf_acc * 100}%\n'
    text.insert(END, result_text)

def knn():
    global knn_acc
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for K-Nearest Neighbors is {knn_acc * 100}%\n'
    text.insert(END, result_text)

def gnb():
    global gnb_acc
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    gnb_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for Gaussian Naïve Bayes is {gnb_acc * 100}%\n'
    text.insert(END, result_text)

def lr():
    global lr_acc
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for Logistic Regression is {lr_acc * 100}%\n'
    text.insert(END, result_text)

def svm():
    global svm_acc
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for Support Vector Machine is {svm_acc * 100}%\n'
    text.insert(END, result_text)

def lda():
    global lda_acc
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    lda_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for Linear Discriminant Analysis is {lda_acc * 100}%\n'
    text.insert(END, result_text)

def plot_results():
    global ada_acc, rf_acc, knn_acc, gnb_acc, lr_acc, svm_acc, lda_acc

    algorithms = ['AdaBoost', 'Random Forest', 'KNN', 'Gaussian NB', 'Logistic Regression', 'SVM', 'LDA']
    accuracies = [ada_acc * 100, rf_acc * 100, knn_acc * 100, gnb_acc * 100, lr_acc * 100, svm_acc * 100, lda_acc * 100]
    explode = (0.1, 0, 0, 0, 0, 0, 0)  # Explode first slice (AdaBoost)

    # Colors for each slice
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0', '#ffb3e6', '#c2f0c2']

    plt.figure(figsize=(8, 6))
    plt.pie(accuracies, explode=explode, labels=algorithms, colors=colors, autopct='%1.1f%%', startangle=140, shadow=True)
    plt.title('Accuracy of Machine Learning Algorithms')

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')

    # Add legend
    plt.legend(algorithms, loc="best", title="Algorithms")

    plt.show()




def predict():
    global text
    
    # Open file manager to select CSV file
    filename = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

    if filename:
        # Read the selected CSV file
        input_data = pd.read_csv(filename)

        # Fill missing values with mode for each column (assuming similar preprocessing as done in other functions)
        input_data.fillna(input_data.mode().iloc[0], inplace=True)

        # Preprocess input data (if needed)
        label_encoder = LabelEncoder()
        for column in input_data.columns:
            if input_data[column].dtype == 'object':
                input_data[column] = label_encoder.fit_transform(input_data[column])

        # Perform prediction using the Random Forest model
        y_pred = rf.predict(input_data)

        # Display the prediction result in the text box
        result_text = "Prediction Results:\n"
        for i, prediction in enumerate(y_pred):
            if prediction == 1:
                result_text += f"Row {i + 1}: Autism Spectrum Disorder Detected\n"
            else:
                result_text += f"Row {i + 1}: Autism Spectrum Disorders Not Detected\n"

        # Clear the text box and display the prediction results
        text.delete('1.0', END)
        text.insert(END, result_text)

        # Show a message box indicating the prediction results
        messagebox.showinfo("Prediction Results", "Prediction completed and displayed in the text box.")


main = tk.Tk()
main.title("A Machine Learning Framework for Early-Stage  Detection of Autism Spectrum Disorders") 
main.geometry("1600x900")

font = ('times', 16, 'bold')
title = tk.Label(main, text='A Machine Learning Framework for Early-Stage Detection of Autism Spectrum Disorders',font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)           
title.config(height=3, width=145)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = tk.Text(main, height=20, width=180)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
button_bg_color = "lightgrey"
button_fg_color = "black"
button_hover_bg_color = "grey"
button_hover_fg_color = "white"
bg_color = "#32d1a7"  # Light blue-green background color

# Define button configurations
button_config = {
    "bg": button_bg_color,
    "fg": button_fg_color,
    "activebackground": button_hover_bg_color,
    "activeforeground": button_hover_fg_color,
    "width": 15,
    "font": font1
}

uploadButton = tk.Button(main, text="Upload Dataset", command=upload, **button_config)
pathlabel = tk.Label(main)
splitButton = tk.Button(main, text="Split Dataset", command=splitdataset, **button_config)
adaboostButton = tk.Button(main, text="AdaBoost (AB)", command=adaboost, **button_config)
rfButton = tk.Button(main, text="Random Forest (RF)", command=random_forest, **button_config)
knnButton = tk.Button(main, text="K-Nearest Neighbors (KNN)", command=knn, **button_config)
gnbButton = tk.Button(main, text="Gaussian Naïve Bayes (GNB)", command=gnb, **button_config)
lrButton = tk.Button(main, text="Logistic Regression (LR)", command=lr, **button_config)
svmButton = tk.Button(main, text="Support Vector Machine (SVM)", command=svm, **button_config)
ldaButton = tk.Button(main, text="Linear Discriminant Analysis (LDA)", command=lda, **button_config)
plotButton = tk.Button(main, text="Plot Results", command=plot_results, **button_config)
predict_button = tk.Button(main, text="Prediction", command=predict, **button_config)

uploadButton.place(x=50, y=600)
pathlabel.config(bg='DarkOrange1', fg='white', font=font1)  
pathlabel.place(x=250, y=600)
splitButton.place(x=450, y=600)
adaboostButton.place(x=50, y=650)
rfButton.place(x=250, y=650)
knnButton.place(x=450, y=650)
gnbButton.place(x=650, y=650)
lrButton.place(x=850, y=650)
svmButton.place(x=1050, y=650)
ldaButton.place(x=50, y=700)
plotButton.place(x=450, y=700)
predict_button.place(x=650, y=700)

main.config(bg=bg_color)
main.mainloop()

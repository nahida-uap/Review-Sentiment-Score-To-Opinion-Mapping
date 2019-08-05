#!/usr/bin/env python
# @author Nahida Sultana Chowdhury <nschowdh@iu.edu>

from pandas import DataFrame
from sklearn import linear_model
import tkinter as tk 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

SSA2SL = {'polarity': [1, 0.75, 0.5, 0.25, -0.25, -0.5, -0.75, -1],
                'subjectivity': [1, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 1],
                 'belief': [1, 0.75, 0.5, 0.25, 0, 0, 0, 0],
                 'disbelief': [0, 0, 0, 0, -0.25, -0.5, -0.75, -1]
                 }

df = DataFrame(SSA2SL,columns=['polarity','subjectivity','belief', 'disbelief', 'uncertainity'])


X = df[['polarity','subjectivity']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['polarity'] for example.Alternatively, you may add additional variables within the brackets
Y = df['belief']
Z = df['disbelief']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print('Intercept_b: \n', regr.intercept_)
print('Coefficients_b: \n', regr.coef_)

regr_d = linear_model.LinearRegression()
regr_d.fit(X, Z)
print('Intercept_d: \n', regr_d.intercept_)
print('Coefficients_d: \n', regr_d.coef_)

# tkinter GUI
root= tk.Tk()
canvas1 = tk.Canvas(root, width = 500, height = 350)
root.title("Sentiment value to opinion mapping")
canvas1.pack()

# with sklearn - belief
Intercept_result_b = ('Intercept_belief: ', regr.intercept_)
label_Intercept_b = tk.Label(root, text=Intercept_result_b, justify = 'center')
canvas1.create_window(260, 150, window=label_Intercept_b)

Coefficients_result_b  = ('Coefficients_belief: ', regr.coef_)
label_Coefficients_b = tk.Label(root, text=Coefficients_result_b, justify = 'center')
canvas1.create_window(260, 170, window=label_Coefficients_b)

# with sklearn - disbelief
Intercept_result_d = ('Intercept_disbelief: ', regr.intercept_)
label_Intercept_d = tk.Label(root, text=Intercept_result_d, justify = 'center')
canvas1.create_window(260, 190, window=label_Intercept_d)

Coefficients_result_d  = ('Coefficients_disbelief: ', regr.coef_)
label_Coefficients_d = tk.Label(root, text=Coefficients_result_d, justify = 'center')
canvas1.create_window(260, 210, window=label_Coefficients_d)

# New_Polarity label and input box
label1 = tk.Label(root, text='Type Polarity: ')
canvas1.create_window(100, 50, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 50, window=entry1)

# New_Subjectivity label and input box
label2 = tk.Label(root, text='Type Subjectivity: ')
canvas1.create_window(120, 70, window=label2)

entry2 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 70, window=entry2)

def values(): 
    global New_Polarity #our 1st input variable
    New_Polarity = float(entry1.get()) 
    
    global New_Subjectivity #our 2nd input variable
    New_Subjectivity = float(entry2.get()) 
    
    b = regr.predict([[New_Polarity ,New_Subjectivity]])
    if b < 0 :
        b = -1 * b
    Prediction_result_b  = ('Predicted Belief: ', b)
    label_Prediction_b = tk.Label(root, text= Prediction_result_b, bg='yellow')
    canvas1.create_window(260, 250, window=label_Prediction_b)

    d = regr_d.predict([[New_Polarity ,New_Subjectivity]])
    if d < 0 :
        d = -1 * d
    Prediction_result_d  = ('Predicted Disbelief: ', d)    
    label_Prediction_d = tk.Label(root, text= Prediction_result_d, bg='yellow')
    canvas1.create_window(260, 270, window=label_Prediction_d)
    
    #b + d + u = 1
    u = 1 - (b + d)
    Prediction_result_u = ('Predicted Uncertainity: ', u) 
    label_Prediction_u = tk.Label(root, text= Prediction_result_u, bg='yellow')
    canvas1.create_window(260, 290, window=label_Prediction_u)

    total_trust_score = ('Total Trust Score: ', b + d + u) 
    label_Prediction_trust = tk.Label(root, text= total_trust_score, bg='yellow')
    canvas1.create_window(260, 310, window=label_Prediction_trust)
    
button1 = tk.Button (root, text='Predict Trust',command=values, bg='yellow') # button to call the 'values' command above 
canvas1.create_window(270, 110, window=button1)

#plot 1st scatter 
figure3 = plt.Figure(figsize=(5,4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(df['polarity'].astype(float),df['belief'].astype(float), color = 'r')
scatter3 = FigureCanvasTkAgg(figure3, root) 
scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax3.legend() 
ax3.set_xlabel('Polarity')
ax3.set_title('Polarity Vs. Belief')


#plot 2nd scatter 
figure4 = plt.Figure(figsize=(5,4), dpi=100)
ax4 = figure4.add_subplot(111)
ax4.scatter(df['polarity'].astype(float),df['disbelief'].astype(float), color = 'blue')
scatter4 = FigureCanvasTkAgg(figure4, root) 
scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax4.legend() 
ax4.set_xlabel('Polarity')
ax4.set_title('Polarity Vs. Disbelief')

root.mainloop()

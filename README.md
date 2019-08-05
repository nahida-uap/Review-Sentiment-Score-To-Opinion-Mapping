# Review-Sentiment-Score-To-Opinion-Mapping

As input user will provide the Sentiment score of the review, in terms of Polarity [-1 to +1] and Subjectivity [0 to 1].

Then, the system will map the sentiment value to opinion model, which is a tupple of belief(b), disbelief(d) and uncertainity(u) where b + d + u =1.

The approach is based on Multiple Linear Regression and implemented in Python2.7+.

To execute the program you need to import the following libraries:

- pandas
- sklearn
- tkinter
- matplotlib


To run the code type the following command:

  - python SSA2SL.py


Application GUI:
![SS2SL](https://raw.githubusercontent.com/nahida-uap/Review-Sentiment-Score-To-Opinion-Mapping/master/img/GUI.png)

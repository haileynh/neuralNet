# Styles & Signs Final Challenge Starter Code

This is a place for you to start building your Styles and Signs Final Challenge Project!

### Project Requirements
Your project should:
- Use the correct (SignMNIST) training and test datasets. 
- Change the number and/or the split of the data between validation and training datasets. 
- Change the design of the neural net by changing the size of the hidden layer.
- Contain a print statement which displays a measure of the final accuracy. 
- Contain print statements indicating which two or three letters the model most often misidentifies.



### Extensions
You can extend your project further by:
- Trying a different ML technique
- Exploring Random Variability



###  Attributions
*If you used any code, stories, or poems from another person or group of people, tell us about it here. Make sure it is in the public domain, has a license that allows you to use it, or is one of your own. 
- Sign MNIST dataset source: https://www.openml.org/search?type=data&status=active&id=45082

---

## File Overview

### ← README.md

README.md file give you more documentation and information about a program. They are super helpful for describing what a program should do, any issues you've encountered, changes you want to make, and more. 

### ← main.py
This is where you will write your main program.

### ← ClassViewer.py
This utility program displays the first 10 images of the class `class_number`.
You can run the program by typing `python ClassViewer.py` in the Shell tab.

### ← DataDumper.py
This utility program displays the first 5 and last 5 rows  and the first two and last two columns of the SignMNIST data set.
You can run the program by typing `python ClassViewer.py` in the Shell tab.


### ← DisplayClassHist.py
This utility program displays a histogram of the SignMNIST data set classes.
You can run the program by typing `python DisplayClassHist.py` in the Shell tab.

### ← ShowPixelGrid.py
This utility program displays a one image from the SignMNIST data set classes with the numerical pixel values overlayed  on each pixel. You can select which row of the dataset is displayed by changing the row number in the iloc location: `row = df.iloc[7]`
You can run the program by typing `python DisplayClassHist.py` in the Shell tab.

### ← sign_mnist_13bal_test.csv
This is a comma separated (CSV) formatted file containing a test portion from a balanced sample of 13 images from each class in the SignMNIST dataset.

### ← sign_mnist_13bal_train.csv
This is a comma separated (CSV) formatted file containing a training portion a balanced sample of 13 images from each class in the SignMNIST dataset.
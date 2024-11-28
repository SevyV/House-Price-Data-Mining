# **House Prices: Data Mining Techniques**
An analysis of a variety of data mining algorithms using House Prices data.

## **Table of Contents**
1. [Installation](#installation)
2. [Usage](#usage)
3. [Running the Code](#running-the-code)


   
## **Installation**
1. Ensure you have python 3.11.4 installed (or another version but this one was used during development). Otherwise, install it.
2. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repository.git

3. Navigate into the project directory:
   ```bash
   cd House-Price-Data-Mining

4. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib


## **Usage**

Running the program with default settings will run all the algorithms without utilizing any of the implemented feature selection algorithms. 
To use them you'll want to navigate to line 59 in main.py and comment out the following lines of code:

    X = data.drop(columns=["PriceCategory"])
    y = data["PriceCategory"]
    
And then comment in either one of these lines of code which are directly above the two aforementioned lines:

    # X, y = preprocessor.feature_selection(data) (This is the mutual information feature selection function)
    # X, y =  preprocessor.lasso_feature_selection(data)   (This is the lasso regression feature selection function)
    
Once that is done just run the program as before, but this time the algorithms will be taking advantage of feature selection. 

## **Running the Code**
In order to run the code you'll want need to run the main.py file directly. The dataset is included so if all of that is installed, no extra command line arguments are necessary to execute the code.
```bash
python main.py









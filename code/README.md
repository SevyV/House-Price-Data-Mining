### Installations

### Usage

Running the program with default settings will run all the algorithms without utilizing any of the implemented feature selection algorithms. 
To use them you'll want to navigate to line 59 in main.py and comment out the following lines of code:
    X = data.drop(columns=["PriceCategory"])
    y = data["PriceCategory"]
And then comment in the either one of these lines of code which are directly above the two aforementioned lines:
    # X, y = preprocessor.feature_selection(data) (This is the mutual information feature selection function)
    # X, y =  preprocessor.lasso_feature_selection(data)   (This is the lasso regression feature selection function)
    
Once that is done just run the program as before, but this time the algorithms will be taking advantage of feature selection. 

### Running the Code

### Example

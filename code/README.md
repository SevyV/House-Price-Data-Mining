### Installations

You can install the files directly off of the repository, the dataset will be included in that installation along with an accompanying txt file describing the various features within. 

### Usage

Running the program with default settings will run all the algorithms without utilizing any of the implemented feature selection algorithms. 
To use them you'll want to navigate to line 59 in main.py and comment out the following lines of code:

    X = data.drop(columns=["PriceCategory"])
    y = data["PriceCategory"]
    
And then comment in either one of these lines of code which are directly above the two aforementioned lines:

    # X, y = preprocessor.feature_selection(data) (This is the mutual information feature selection function)
    # X, y =  preprocessor.lasso_feature_selection(data)   (This is the lasso regression feature selection function)
    
Once that is done just run the program as before, but this time the algorithms will be taking advantage of feature selection. 

### Running the Code

In order to run the code you'll want need to run the main.py file directly. The dataset is included so if all of that is installed, no extra command line arguments are necessary to execute the code.
However, if you have the data at a different location on your computer, you can manually input the file paths via the command line like so: 

### Example

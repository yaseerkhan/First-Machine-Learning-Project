# First-Machine-Learning-Project

 Machine Learning With Python

Environment Used : Jupyter
Algorithm used : DecisionTree

When working with machine learning projects we use an environment called jupyter for writing our code. technically we can still use VScode or any other code editors but these editors are not idle for machine learning projects because we frequetly need to inspect the data and that is really hard in envs like VScode and terminal, if we are working with a table of 10 or 20 cols visualising these data in a terminal window is really diffcult and messy so thats why we use jupyter. It makes it really easy to inspect our data

Libraries- 
1. Numpy (Very popular libraly): Provides a multi-dimensional array 
2. Pandas (Very popular in machine learning and datascience projects) : Data Analysis library that provide a concept called DataFrame. A DataFrame is a 2-dimensional data structure similar to an excel spreadsheet, so we have rows and columns, we can select data in a row or column or range of rows and columns
3. MatplotLib : A 2-dimensional plotting library for creating graphs and plots
4. SciKit-Learn(one of the most popular machine learning libraries) : Provides all the common algorithms like Decision-Tree, Neural-Network and so on



Steps to install Jupyter:
1. Go to https://www.anaconda.com/products/distribution and download Anaconda Distribution. It includes jupyter and other tools to work with.
2. After installing Anaconda, Launch it and launch jupyter from there OR open anaconda prompt/terminal -> type "jupyter notebook" it will launch jupyter.
3. We will get jupyter dashboard , from there navigate to where ever you want to place the project. In my case i created a folder on desktop and navigated there. 
4. Import a dataset from .csv file in jupyter, for that download a dataset from a very popular site https://kaggle.com , after visiting we have to signup to download anything, Search for video game sales very first result would be from gregory smith with some redish thumbnail download that dataset and place it in the project folder.

5. Go back to jupyter notebook and code :

    import pandas as pd
    # This returns a dataframe object like an excel spreadsheet, storing it in df object.
    df = pd.read_csv('vgsales.csv') # This dataframe object has lots of attributes and methods, read pandas docs for more.

    # We will look at some most useful attributes and methods. () at end represents methods and without () means attributes.

    df.shape # returns rows and cols count
    df.describe() # returns basic information like count, mean, standard deviation, min value, 25% , 50%, 75%, max value
    df.values # returns 2-Dimensional array representation of dataset


Now working with actual project, Algorithm/Library used for current project - SciKit-Learn

file- music.csv : it has some random information which have cols: age, gender and genre.
input = training set, output = dataset.

How to 1. prepare the data and 2. clean the data such as removing duplicates

1. Preparing data - splitting dataset in 2 seprate datasets, One with first 2 cols which to refer as input-set and the other with last col i.e genre which we refer to as output-set, the output-set contains the predictions so we are training our model that if we have a user whos 20yrs old and hes a male they likes hiphop once we train our model we give it a new input set, we say hey we have a new user whos 21 years old and hes a male what is the genre of the music this user will like as we can see from the table we dont have a sample for 21 year old male so we will ask the model to predict that that is the reason we need to split the dataset into 2 separate set input and output.

    import pandas as pd
    music_data = pd.read_csv('music.csv')
    # .drop will not modify the original dataset but it will created a new dataset without the genre column. This is input dataset
    X =  music_data.drop(columns=['genre'])
    # here we stored the predictions in lowercase y var. This is our output-dataset
    y = music_data['genre']

## Learning and Predicting
    # Now we build a model using machine learning algorithm, we will use a very simple algorithm decision tree. 
    # It is already implemented in a libraby called SciKit-Learn.

    import pandas as pd
    # Here we imported the algorithim of DecisionTree 
    from sklearn.tree import DecisionTreeClassifier
    
        music_data = pd.read_csv('music.csv')

    # .drop will not modify the original dataset but it will created a new dataset without the genre column. This is input dataset
        X =  music_data.drop(columns=['genre'])
    
    # here we stored the predictions in lowercase y var. This is our output-dataset
        y = music_data['genre']

    # now create an object and set it to a new instance of DecisionTreeClassifier()
        model = DecisionTreeClassifier()

    # giving our model the datasets to train it so it learns pattern in data. .fit() method take 2 datasets i.e input & output but i read on  stack that its not necessary unless the loaded values are correspoding. The Comment was :"Sci-kit-Learn needs i/p (training set), and o/p data set as arguments when fitting them. Inserting these as data-frame with header/feature or just values (i.e., multi-dimensional arrays) will NOT make any difference. The difference might occur only if the features selected when loading the values are not corresponding to the specific training or output/testing data set"
        model.fit(X.values, y) # using only X gives a warning so i used X.values. In the actual course he used X.

    # .predict() method take the 2-d array looks like this: model.predict([ [] ]), in this array each element is on array, here we are asking our model to make two predictions at the same time. single prediction would look like model.predict([ [21, 1] ])
        predictions = model.predict([ [21, 1], [20, 0] ])

## Calculating the Accuracy : In order to do so first we have to split our data into 2 sets , 1 for training and other for testing because rightnow we are passing the entire dataset for training the model and we are using 2 samples for making predictions that is not enough to calculate the accuracy of the model. In general rule of thumb it is to allocate 70-80% of data for training and other 20-30% for testing then instead of passing only 2 samples for making predictions we can pass the dataset we have for testing will get the predictions and then we can compare this predictions  with the actual values in the test set based on that we can calculate the accuracy

    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    
    # This will help us to split data into 2 dataset to calculate accuracy
    from sklearn.model_selection import train_test_split
    
    # will calculate accuracy using this
    from sklearn.metrics import accuracy_score

    music_data = pd.read_csv('music.csv')
    X = music_data.drop(columns=['genre'])
    y = music_data['genre']
    
    # using train_test_split() function this will randomly select data from dataset, test_size=0.2 is a keyword argument that spcifies
    # the size of our test datasets so we are allocating 20%(0.2) of our data for testing. This function(train_test_split) returns a tuple
    # so we can unpack it into four variables, so the first 2 variables are input-sets for training and testing and last 2 variables
    # are output-sets for training and testing
    X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2)

    model = DecisionTreeClassifier()
    
    # now when training our model instead of passing the entire dataset we pass the training dataset
    # model.fit(X.values, y) - This is our old line of code when we were passing the entire dataset
    model.fit(X_train, y_train)
    
    # Also when making predictions instead of passing 2 samples like we did earlier , we pass test-set "X_test" This is the dataset
    # which includes dataset for testing
    predictions = model.predict(X_test)
    predictions

    # y_test contains the expected values and predictions which contains the actual values, This function returns the
    # accuracy score between 0-1
    score = accuracy_score(y_test, predictions)
    score


## Persisting Models : 

    
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    import joblib
    # from sklearn.externals import joblib - This was not working for me. Maybe updated the path

    # This piece of code is not what we wanna run everytime we have new user or everytime we wanna
    # make recommendation for an existance user because training a model can sometimes be very time
    # consuming. In this example are dealing with a very small dataset that has only 20 records
    # but in real applications we might have a dataset with thousands or millions of samples training a 
    # model for that take seconds or min or even hours so that is why model persistance is imp.
    # Once in a while we build and train our model and then we'll save it to a file now next time we wanna
    # make predictions we simply import the model from the file and ask it to make predictions.
    # That model is already trained we dont need to re-train it. It's like an intelligent person.
   
    # BLOCK OF CODE WE WERE TALKING ABOUT STARTS HERE. THIS BLOCK GETS STORED IN A FILE USING JOBLIB --------

        # This block is importing dataset
        music_data = pd.read_csv('music.csv')
        X = music_data.drop(columns=['genre'])
        y = music_data['genre']

        # Creating our model
        model = DecisionTreeClassifier()
        
        # Training our model
        model.fit(X.values, y)
    
    # END HERE ---------------------------------------------

    # This stores our trained model into a file.
    joblib.dump(model, 'music-recommender.joblib')

    # Asking it to make prediction
    # predictions = model.predict([ [21, 1] ])
    # predictions

    # As of now we stored our training model into a file now we will be importing that file
    # and directly making predictions using that file so we dont have train our model again
    # and again. we only use joblib.dump to save file after that comment it out.

    # To load our data from the file
    model = joblib.load('music-recommender.joblib')

    # now we will make predictions
    predictions = model.predict([ [21, 1] ])
    predictions


## Visualizing a Decision Tree : 

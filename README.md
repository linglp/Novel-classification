# Novel-classification

## Problems to solve:
I tried to build a model to classify sentences from two novels: A tale of Two Cities and King Lear. 

## Step 1: read content
A tale of Two Cities is a novel written by Charles Dickens, and King Lear is a play written by William Shapespear. As we could see, the format of these two are very different. For example, play has character names, dialogue, and act/scene designations while novel has relatively longer sentences. 

When I read content from raw data, I also tried to make sure the sentences contain relatively the same amount of "juice" (aka words other than stop words)

## Step 2: Use feature extraction functions from Sklearn to turn words to vectors
Bag of words is a way to extract features from text for building machine-learning models. For example, we could extract all the unique words from a list of sentences and construct a table with frequency of each word. For example, if we have the following two sentences:
1) This is the best of times.
2) This is the worst of times.

We could extract all the unique words like this: 
* this
* is
* best
* times
* worst

For the first sentence, "this is the best of times" = [1, 1, 1, 1,0]. For the second sentence, "this is the worst of times" = [1, 1, 0, 1, 0]

SKlearn has CountVectorizer class and TfidfVectorizer class to transform words to vectors. 

Source: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

## Step 3: Model building and evaluation 
* GaussianNB: has a testing accuracy rate of 0.85
* Multinomial model: has a testing accuracy rate of 0.87
* ComplementNB: has a testing accuracy rate of 0.83
* Decision tree: has a testing accuracy rate of 0.83 (has a problem of overfitting)
* Random forest: 
1) Initially, the model has an accuracy rate of 0.8
2) But after finding the best params by using RandomizedSearchCV from Sklearn, the model accuracy rate becomes 0.89 for testing and 0.98 for training. 


# Applying of neural network technologies to word segmentation of Сhinese hierogliphic texts

_The project is dedicated to study on the effectiveness of one-layer neural network for meeting the challenge of word segmentation of hierogliphic Chinese texts for [Russian-Chinese parallel corpus](http://www.ruscorpora.ru/search-para-zh.html)_.

---

### The repository includes:

* [**A base of training data**](https://github.com/leramorozova/WordSplitter/blob/master/characters.db)

The base consists of two tables: the list of unique characters in training data and training data list itself. Training data vertorization has been performed with the use of standart scaling.
The data has been compiled from starter educational sources to produce artificial restricrion of vocabulary diversity. Training units are hierogliphic sentences splitted by punctuation.

* [**A scrypt performing preparation of dataset**](https://github.com/leramorozova/WordSplitter/blob/master/dataset_maker.py)

The scrypt converts text files to dataset which is ready for using in training and testing procedures.

* [**A scrypt perfoming construction of a NN**](https://github.com/leramorozova/WordSplitter/blob/master/main.py)

There is a class of neural network, which includes methods of training and cross validation. In addition, there are functions for validation of NN hyperparametres.

* [**A scrypt perfoming plot visualisation**](https://github.com/leramorozova/WordSplitter/blob/master/plotting.py)

The one was used while searching optimal hyperparametres for NN.

* [**A collection of plots**](https://github.com/leramorozova/WordSplitter/tree/master/plots)

The product of the scrypt mentioned above.

* [**Collection of sentences splitted by punctuation**](https://github.com/leramorozova/WordSplitter/blob/master/train.txt)

The texts collected from textboox and educational web-sources for starter learners.

* [**Collection of spaced sentences**](https://github.com/leramorozova/WordSplitter/blob/master/train_spaces.txt)

The same texts segmented manually in order to train the NN.

---

### NN configurations:

**Database size:** 875 units

**Amount of cv-blocks:** 7 (125 units per block)

**Amount of nodes in hidden layer:** 550

**Amount of the training cycles:** 270

**Learning rate:** 0,18

The values were estimated experimentally.

___

### The text of course project:

coming soon



 

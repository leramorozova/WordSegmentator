# Applying of neural network technologies to word splitting of Сhinese texts

_The project is dedicated to study on the effectiveness of one-layer neural network for meeting the challenge of word-splitting of hierogliphic Chinese texts for [Russian-Chinese parallel corpus](http://www.ruscorpora.ru/search-para-zh.html)_

---

### The repository includes:

* [**A base of training data**]()
The base consists of two tables: the list of unique characters in training data and training data list itself. Training data vertorization has been performed with the use of standart scaling.
The data has been compiled from starter educational sourses to produce artificial estricrion of vocabulary diversity. Training units are hierogliphic sentences splitted by punctuation.

* [**A scrypt performing preparation of dataset**]()

The scrypt converts text files to dataset which is ready for using in training and testing procedures.

* [**A scrypt perfoming creation of a NN**]()

There is a class of neural network, which includes methods of training ang cross validation. In addition, there are functions for validating hyperparametres of NN.

* [**A scrypt perfoming plot visualisation**]()

That was used during the search of optimal hyperparametres for NN.

* [**A collection of plots**]()

The product of the scrypt mentioned above.


### NN configurations:

**Database size:** 875 units

**Amount of cv-blocks:** 7 (125 units per block)

**Amount of nodes in hidden layer:** 550

**Amount of the training cycles:** 270

**Learning rate:** 0,18

The values were estimated experimentally.

___

#### The text of course project:

coming soon



 

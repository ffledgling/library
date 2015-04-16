What is this about?
===================

This repository 'library' is an implementation of a Tree-based classifier.
This classifier uses linear classifiers at each step to partition the training classes into two sets
or "partitions". These partitions are then subsequently sub-divided until each partition is simply a
singleton set.

The advantages of a tree based classifier such as this is that although it's slow to train, the
classification process is faster than the usual Linear SVM with a `One vs. One` or `One vs. Rest`
classification.  
This classifier requires b/w O(logN) (*best case*) and O(N) (*worst case*) checks to classify a
class against the O(N) number of checks required by both `One vs. One` and `One vs. Rest`

The total number of linear classifiers that need to be remembered is also much smaller than the
number required by `One vs. Rest` classifier, O(N) against the O(<sup>N</sup>C<sub>2</sub>) required
by the `One vs. Rest` classifier.


Using the library
-----------------

Using the library is fairly straight forward, simply pass the training and test files to the main.py
file. This will begin the training process, and will finally output a JSON object that contains the
tree and linear classifiers. The format of this JSON Object is explained in the **JSON Object**
section below.

This repository also contains a lot of code for visualisation of the tree returned by python code.
You can also view the tree being built in real-time in exactly the same way. For more details on how
to do this, please the **Visualisation** section below.

Installing and Using the library
--------------------------------

The library is written in Python and as a result should be fairly easy to use across operating
systems and platforms.
This library was developed against Python 2.7 and should ideally be run as such.
The author was able to get it to run against Python 2.6 with some slight changes to the `set`
initialization and modifications to the `print` statement syntax, but these changes have not been
checked in.

It is recommended that users of the library setup a virtual environment and use the library from
within the `virtualenv`.
To setup a Virtual Enviroment simply use the `virtualenv` command like so:
```
virtualenv library && cd $_ && source bin/activate
```

The packages to install for development, modifications and so forth for the
library code are encapsulated in the `requirements.txt` and can be installed via the following
command (from within an activated virtualenv):
```
pip install -r requirements.txt
```

### Emergency Measures

If everything else fails, please install the following libraries: `numpy`, `scipy` and
`scikit-learn` system wide using your operating system's package manager. Try to avoid this if you
can please.


File Format
-----------

The library takes the train and testing input files in a particular format as explained below:

1. Both files must be CSV files, ***wihtout*** the column headers.
2. Each sample must be on a new line.
3. Number of features in all the samples must be of equal length.
4. On each line, the features must come first, the class label should be the *last* entry on the
   line.

```
feature1, feature2, feature3, ..., featureN, class-label #1
feature1, feature2, feature3, ..., featureN, class-label #2
...
```

Please the included `optdigits/optdigits.tra` and `optdigits/optdigits.tra` files for examples.
You can also use these files to check if everything is setup up and working correctly, this dataset
is small and easy to run on, and gives a decent accuracy.


JSON Object format
------------------

The JSON Object format is as follows:
```
{
    "table": {
        // This table is global for all nodes
        "<numeric hash identifying a unique linear classifier>" : [
            N length array defining the co-efficients of the separating hyperplane
        ],
        ...
    }
    "tree": {
        // This is the actual tree structure
        "hash" : "<The hash value used to lookup the classifier in the table>"
        "classes": [ Array of class labels that this node deals with ]
        "negative": <Sub-Tree for samples on the negative side of the classifier>
        "positive": <Sub-Tree for samples on the positive side of the classifier>
    }
}
```

Visualisation
-------------

Everytime the library code is run to train a classifier, it logs it's progress in the file
`live.json`, this filename is configurable via `config.py`, but it is recommended to keep it as is.
This `live.json` is soft-linked from within the `vis/` folder by the same name.

All the visualisation magic happens within this folder. Simply start an HTTP Server from within this
folder to start serving the files, you can do so by typing the following in your terminal from
within the `vis/` folder:
```
python -m SimpleHTTPServer
```

Then simply point your browser to `localhost:8000` to look at the progress.
The tree is continuously updated every 5 seconds, but the actual reflected depends on the data logged
(which is infrequent and happens when it does...), therefore smaller datasets may update in large
chunks, whereas the larger ones may take quite a bit of time to show anything at all. In such case,
patience is your ally.

Caveat: The updates and polling begin to hog memory after a while, so hard refresh (Ctrl+Shift+R in
most browsers) can help if that happens. The tree will resume from wherever it was being built, so
do not worry about losing progress.

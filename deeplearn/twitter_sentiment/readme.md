# ***Neural networks***

## **What is needed**

### *Preprocessing the training files*

For sentiment analysis, most datasets are in the form of CSV files which contains a line of text, and the sentiment associated with it.  In order to keep accurate statistics during training, we need to know the number of samples in advance, so that given a batch size, we can calculate the number of batcher per epoch and output the progress inside an epoch along with the relevant statistics.  To speed training up, we shouldn't have to read through the entire dataset more than once.

The **twitter sentiment dataset** is rather, but it fits easily in memory.  The data reader for such a file can therefore read through the file once, and save the data it reads to a buffer in memory which it can use for later epochs.

We can write a generic preprocessing script for csv files which reads through the file, and writes all the data to an *SQL* database.  The database can have one table for each source.  No preprocessing is done on the strings before entering the database.  The database can also contain tables for other metadata, such as word frequency.  the database can enforce character encoding.

### *Batch generation*

Given an archive containing the dataset's metadata and the actual data, we generate batches of size N units, where a unit is defined to be a row in the database. Before being batched together, we run a text preprocessor (or not).  The batch generators keeps track of the number of batches per epoch, and the total number of batches that it will generate.

#### Structure

One process reads from the database, one process does the preprocessing (padding and truncating) and one process procudes the batches. Using python *multiprocessing* package to get around the GIL, these should run on separate cores.  The character emedding and the one-hot encoding of the output is computed by the model. This way batch generators can be encoding independent.

Reading from the database can be done in chunks, and the size of the chunk should be a parameter passed to the reader.  The data reader feeds a stream of dictionaries to the batch generator.  As soon as the batch generator has enough data, it emits a batch

#### Language Independence

We should use byte-level encoding in order for the model to be language independent.  With byte-level encoding, the characters are encoded in utf-8.  The morphology of the characters themselves can be captured using byte-level encoding. Also, unicode has codepoints for emoticons and other characters which can alter the sentiment of a message. There are no such characters in the twitter sentiment dataset, so as

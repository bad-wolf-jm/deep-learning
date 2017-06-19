# Neural networks

## What is needed

### Preprocessing the training files

For sentiment analysis, most datasets are in the form of CSV files which contains a line of text, and the sentiment associated with it.  In order to keep accurate statistics during training, we need to know the number of samples in advance, so that given a batch size, we can calculate the number of batcher per epoch and output the progress inside an epoch along with the relevant statistics.  To speed training up, we shouldn't have to read through the entire dataset more than once.

The **twitter sentiment dataset** is rather, but it fits easily in memory.  The data reader for such a file can therefore read through the file once, and save the data it reads to a buffer in memory which it can use for later epochs.

*We can write a generic preprocessing script for csv files which reads through the file, and counts the number of lines, as well as the number of instances of each category we want to train for.  The CSV file, as well as the metadata can be saved to a zip file, of a tar.gz archive (python can read those like normal files and folders).*

We can write a generic preprocessing script for csv files which reads through the file, and writes all the data to an *sql* database

### Batch generation

Given an archive containing the dataset's metadata and the actual data, we generate batches of size N units, where a unit is defined to be either a line of text, or a blob of a predefined size (images, for example)

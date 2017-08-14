import copy
from train.data import vader_sentiment_training_generator
from train.data import sentiwordnet_sentiment_training_generator
from train.data import cms_training_generator
from train.data import user_cms_training_generator
from train.data import sstb3_training_generator
from train.data import sstb5_training_generator


generator_specs = {
    "BuzzometerDatasetVader": {
        'name': "BuzzometerDatasetVader",
        'display_name': "Buzzometer Dataset (VADER output)",
        'description': 'Sentiment data from the relevant table of the Buzzometer database. '
                        'The sentiment values for this dataset is taken from the output of '
                        'the VADER sentiement classifier',
        'type': 'categorical_data',
        'language': 'en',
        'constructor': vader_sentiment_training_generator,
        'number_of_classes': 3,
        'category_labels': {0:'Negative',
                            1:'Neutral',
                            2:'Positive'}
    },

    "BuzzometerDatasetSentiwordnet": {
        'name': "BuzzometerDatasetSentiwordnet",
        'display_name': "Buzzometer Dataset (SENTIWORDNET output)",
        'description': 'Sentiment data from the relevant table of the Buzzometer database. '
                        'The sentiment values for this dataset is taken from the output of '
                        'the SENTIWORDNET sentiement classifier',
        'type': 'categorical_data',
        'language': 'en',
        'constructor': sentiwordnet_sentiment_training_generator,
        'number_of_classes': 3,
        'category_labels': {0:'Negative',
                            1:'Neutral',
                            2:'Positive'}
    },

    "CMSDataset": {
        'name': "CMSDataset",
        'display_name': "CMS Flagging Dataset",
        'description': 'Flagging data from the CMS comment flagging system. The comments are flagged according '
                        'to their general topic.  This dataset consists of the five most popular categories in '
                        'the database.',
        'type': 'categorical_data',
        'language': 'en',
        'constructor': cms_training_generator,
        'number_of_classes': 5,
        'category_labels': {0:'Bug',
                            1:'Question',
                            2:'Suggestion',
                            3:'Positive feedback',
                            4:'Negative feedback'}
    },

    "CMSUserInputDataset": {
        'name': "CMSUserInputDataset",
        'display_name': "User Input and CMS Flagging Dataset",
        'description': 'Flagging data from the CMS comment flagging system. The comments are flagged according '
                        'to their general topic.  This dataset consists of the five most popular categories in '
                        'the database.',
        'type': 'categorical_data',
        'language': 'en',
        'constructor': user_cms_training_generator,
        'number_of_classes': 4,
        'category_labels': {0:'Negative',
                            1:'Neutral',
                            2:'Positive',
                            3:'Irrelevant'}
    },



    "SSTBDataset_3": {
        'name': "SSTBDataset_3",
        'display_name': "Stanford Sentiment Treebank dataset (3 categories)",
        'description': 'This is the dataset of the paper: Recursive Deep Models for Semantic Compositionality '
                        'Over a Sentiment Treebank Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, '
                        'Christopher Manning, Andrew Ng and Christopher Potts Conference on Empirical Methods '
                        'in Natural Language Processing (EMNLP 2013). This dataset splits the sentiment values '
                        'into three categories: [0, 1/3), [1/3, 2/3), [2/3,1]',
        'type': 'categorical_data',
        'language': 'en',
        'constructor': sstb3_training_generator,
        'number_of_classes': 3,
        'category_labels': {0:'Negative',
                            1:'Neutral',
                            2:'Positive'}
    },

    "SSTBDataset_5": {
        'name': "SSTBDataset_5",
        'display_name': "Stanford Sentiment Treebank dataset (5 categories)",
        'type': 'categorical_data',
        'description': 'This is the dataset of the paper: Recursive Deep Models for Semantic Compositionality '
                        'Over a Sentiment Treebank Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, '
                        'Christopher Manning, Andrew Ng and Christopher Potts Conference on Empirical Methods '
                        'in Natural Language Processing (EMNLP 2013). This dataset splits the sentiment values '
                        'into three categories: [0, 1/5), [1/5, 2/5), [2/5,3/5), [3/5, 4/5), [4/5, 1]',
        'language': 'en',
        'constructor': sstb5_training_generator,
        'number_of_classes': 5,
        'category_labels': {0:'Very Negative',
                            1:'Negative',
                            2:'Neutral',
                            3:'Positive',
                            4:'Very Positive'}
    }
}


def get_dataset_specs(type):
    return copy.deepcopy(generator_specs.get(type, None))

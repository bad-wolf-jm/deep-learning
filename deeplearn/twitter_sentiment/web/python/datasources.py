import copy
from train.data import sentiment_training_generator
from train.data import cms_training_generator

# NOTE: THIS IS INCOMPLETE, coms of the cataset consrtuctors are missing

generator_specs = {
    "BuzzometerDatasetVader": {
        'name': "BuzzometerDatasetVader",
        'display_name': "Buzzometer Dataset (VADER output)",
        'description': 'Sentiment data from the relevant table of the Buzzometer database. '
                        'The sentiment values for this dataset is taken from the output of '
                        'the VADER sentiement classifier',
        'type': 'categorical_data',
        'language': 'en',
        'constructor': sentiment_training_generator,
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
        'constructor': sentiment_training_generator,
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
        'constructor': None,
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
        'constructor': None,
        'number_of_classes': 3,
        'category_labels': {0:'Very Negative',
                            1:'Negative',
                            2:'Neutral',
                            3:'Positive',
                            4:'Very Positive'}
    },

    "BlogGenderDataset": {
        'name': "BlogGenderDataset",
        'display_name': "Gender classification of blog posts",
        'type': 'categorical_data',
        'description': 'The Blog Authorship Corpus consists of the collected posts of 19,320 bloggers gathered '
                        'from blogger.com in August 2004. The corpus incorporates a total of 681,288 posts and '
                        'over 140 million words - or approximately 35 posts and 7250 words per person.  Each blog '
                        'is presented as a separate file, the name of which indicates a blogger id# and the blogger’s '
                        'self-provided gender, age, industry and astrological sign. (All are labeled for gender '
                        'and age but for many, industry and/or sign is marked as unknown.) All bloggers included '
                        'in the corpus fall into one of three age groups: '
                        '          8240 "10s" blogs (ages 13-17), '
                        '          8086 "20s" blogs(ages 23-27) '
                        '          2994 "30s" blogs (ages 33-47). '
                        'For each age group there are an equal number of male and female bloggers.   '
                        'Each blog in the corpus includes at least 200 occurrences of common English words. All formatting '
                        'has been stripped with two exceptions. Individual posts within a single blogger are separated by the '
                        'date of the following post and links within a post are denoted by the label urllink. '
                        'The corpus may be freely used for non-commercial research purposes. Any resulting publications should '
                        'cite the following: J. Schler, M. Koppel, S. Argamon and J. Pennebaker (2006). '
                        'Effects of Age and Gender on Blogging in Proceedings of 2006 AAAI Spring Symposium on '
                        'Computational Approaches for Analyzing Weblogs.',
        'language': 'multi',
        'constructor': None,
        'number_of_classes': 2,
        'category_labels': {0:'Male',
                            1:'Female'}
    },

    "BlogAgeDataset": {
        'name': "BlogAgeDataset",
        'display_name': "Age classification of blog posts",
        'type': 'categorical_data',
        'description': 'The Blog Authorship Corpus consists of the collected posts of 19,320 bloggers gathered '
                        'from blogger.com in August 2004. The corpus incorporates a total of 681,288 posts and '
                        'over 140 million words - or approximately 35 posts and 7250 words per person.  Each blog '
                        'is presented as a separate file, the name of which indicates a blogger id# and the blogger’s '
                        'self-provided gender, age, industry and astrological sign. (All are labeled for gender '
                        'and age but for many, industry and/or sign is marked as unknown.) All bloggers included '
                        'in the corpus fall into one of three age groups: '
                        '          8240 "10s" blogs (ages 13-17), '
                        '          8086 "20s" blogs(ages 23-27) '
                        '          2994 "30s" blogs (ages 33-47). '
                        'For each age group there are an equal number of male and female bloggers.   '
                        'Each blog in the corpus includes at least 200 occurrences of common English words. All formatting '
                        'has been stripped with two exceptions. Individual posts within a single blogger are separated by the '
                        'date of the following post and links within a post are denoted by the label urllink. '
                        'The corpus may be freely used for non-commercial research purposes. Any resulting publications should '
                        'cite the following: J. Schler, M. Koppel, S. Argamon and J. Pennebaker (2006). '
                        'Effects of Age and Gender on Blogging in Proceedings of 2006 AAAI Spring Symposium on '
                        'Computational Approaches for Analyzing Weblogs.',
        'language': 'multi',
        'constructor': None,
        'number_of_classes': 2,
        'category_labels': {0:'Male',
                            1:'Female'}
    }
}


def get_dataset_specs(type):
    return copy.deepcopy(generator_specs.get(type, None))

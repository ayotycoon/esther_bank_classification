import csv
import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from typing_extensions import Unpack


class BasicClassifier:

    training_data_list = []
    category_dict_key = ''
    title_dict_key = ''
    training_cache = None


    def __init__(self,
                 training_data_list: List,
                 category_dict_key:str = 'Map_Name',
                 title_dict_key:str =   'Map_Text'
                 ):
        self.training_data_list = training_data_list
        self.category_dict_key = category_dict_key
        self.title_dict_key = title_dict_key


    def get_training_clf(self) -> tuple[Any, CountVectorizer, Any]:

        training_data_df = pd.DataFrame(self.training_data_list)
        training_data_df['category_id'] = training_data_df[self.category_dict_key].factorize()[0]
        category_id_df = training_data_df[[self.category_dict_key, 'category_id']].drop_duplicates().sort_values('category_id')

        category_to_id = dict(category_id_df.values)
        id_to_category = dict(category_id_df[['category_id', self.category_dict_key]].values)
        # df
        fig = plt.figure(figsize=(8, 6))
        training_data_df.groupby(self.category_dict_key)[self.title_dict_key].count().plot.bar(ylim=0)

        tfidf = TfidfVectorizer(
            sublinear_tf=True,
            min_df=5,
            norm='l2',
            encoding='latin-1',
            ngram_range=(1, 2),
            stop_words='english')
        features = tfidf.fit_transform(training_data_df[self.title_dict_key]).toarray()
        labels = training_data_df.category_id

        # N = 2
        # for Product, category_id in sorted(category_to_id.items()):
        #     features_chi2 = chi2(features, labels == category_id)
        #     indices = np.argsort(features_chi2[0])
        #     feature_names = np.array(tfidf.get_feature_names_out())[indices]
        #     unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        #     bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        #     print("# '{}':".format(Product))
        #     print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        #     print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

        x_train, x_test, y_train, y_test = train_test_split(training_data_df[self.title_dict_key],
                                                            training_data_df[self.category_dict_key],

                                                            random_state=0)
        count_vect = CountVectorizer()
        x_train_counts = count_vect.fit_transform(x_train)
        tfidf_transformer = TfidfTransformer()
        x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
        clf = LogisticRegression().fit(x_train_tfidf, y_train)

        models = [
            RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42,class_weight="balanced"),
            LinearSVC(class_weight="balanced"),
            MultinomialNB(),
            LogisticRegression(random_state=0,class_weight="balanced"),
        ]
        CV = 5
        entries = []
        for model in models:
            model_name = model.__class__.__name__
            accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

        mean = cv_df.groupby('model_name').accuracy.mean()
        print(f"\nmean \n{mean}")

        model = LogisticRegression(class_weight="balanced")
        x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels,
                                                                                         training_data_df.index,
                                                                                         test_size=0.33, random_state=0)
        model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        #
        # conf_mat = confusion_matrix(y_test, y_pred)
        # fig, ax = plt.subplots(figsize=(10, 10))
        #
        # sns.heatmap(conf_mat, annot=True, fmt='d',
        #             xticklabels=category_id_df.Map_Name.values, yticklabels=category_id_df.Map_Name.values)
        # plt.ylabel('Actual')
        # plt.xlabel('Predicted')
        # plt.show()

        pd.options.display.float_format = '{:,.2f}'.format

        return clf, count_vect, category_id_df.values


    def get_prediction(self, **kwargs):
        if self.training_cache != None:
            clf, count_vect, labels = self.training_cache
        else:
            clf, count_vect, labels = self.get_training_clf()
            self.training_cache = clf, count_vect, labels

        if "test_data_csv_url" in kwargs:
            test_data = pd.read_csv(kwargs["test_data_csv_url"], header=0, encoding='latin-1')
        elif "list" in kwargs:
            titles = [s['title'].lower() for s in kwargs["list"]]
            ids = [s['id'] for s in kwargs["list"] if 'id' in s]
            pd_dict = {'title': titles}
            if len(ids) > 0:
                pd_dict['id'] = ids

            test_data = pd.DataFrame(pd_dict)
        else:
            raise Exception("invalid")

        if "id" not in test_data:
            ids = np.random.randint(low=1e9, high=1e10, size=len(test_data))
            test_data["id"] = ids

        predictions = clf.predict(count_vect.transform(list(test_data['title'])))
        probabilities = clf.predict_proba(count_vect.transform(list(test_data['title'])))

        new_probabilities  = np.empty((probabilities.shape[0], probabilities.shape[1]), dtype=object)
        test_data["predicted"] = predictions

        classes = clf.classes_

        for i, row in enumerate(probabilities):
            p = [{'clas': clas, 'prob': prob} for clas, prob in zip(classes, row)]
            p.sort(key=lambda e: float(e['prob']), reverse=True)
            new_probabilities[i] = p


        new_probabilities_list = new_probabilities.tolist()
        test_data["probabilities"] = new_probabilities_list
        return test_data

    def get_prediction_as_list(self, **kwargs):
        return self.get_prediction(**kwargs).to_dict(orient='records')

    def get_prediction_as_dict(self, **kwargs):
        return {obj['id']: obj for obj in self.get_prediction_as_list(**kwargs)}


with open('main/training.csv', mode='r') as file:
    csv_reader = csv.reader(file)

    data_list = [row for row in csv_reader]
    header = data_list[0]
    data_list = [{header[0]: row[0],header[1]: row[1] } for row in data_list[1:]]

print(data_list)
classifierInstance = BasicClassifier(
    data_list,
     'Category', 'Title')
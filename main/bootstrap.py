from main.basic_classifier import classifierInstance


class Bootstrap:
    @staticmethod
    def init():
        input_list=[{ "title":"AROMA COFEE HOUSE","id":"3434343434" }]
        print(classifierInstance.get_prediction_as_list(list=input_list))



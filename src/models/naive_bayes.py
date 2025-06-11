from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import pickle

class Classifier(object):
    def __init__(self, model_type='multinomial'):
        self.model_type = model_type
        self.model = MultinomialNB() if model_type == 'multinomial' else GaussianNB()

    def training(self, features_train, labels_train):
        self.model.fit(features_train, labels_train)
        
    def save_model(self, file_path):
        "Lưu mô hình vào file pk"
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, file_path)

    def load_model(self, file_path):
        "Tải mô hình từ file pk"
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        
    def training_result(self, features_test, labels_test):
        y_true, y_pred = labels_test, self.model.predict(features_test)
        print(classification_report(y_true, y_pred))

        
    def wrong_predictions(self, X_data, labels_test, features_test):
        """
        In ra các dự đoán sai, bao gồm nội dung (content) và nhãn dự đoán/nhãn thực tế.
        X_data: List of dictionaries, mỗi dict có dạng {'label': ..., 'content': ...}
        """
        y_true, y_pred = labels_test, self.model.predict(features_test)
        
        wrong_predictions = []
        for i, (pred, true) in enumerate(zip(y_pred, y_true)):
            if pred != true:
                wrong_predictions.append({
                    'content': ' '.join(NLP(text = X_data[i]['content']).get_words_feature()),
                    'predicted_label': pred,
                    'true_label': true
                })

        if wrong_predictions:
            print("Các dự đoán sai:")
            for item in wrong_predictions:
                #print(f"- Nội dung: {item['content']}")
                print(f"  - Dự đoán: {item['predicted_label']}")
                print(f"  - Thực tế: {item['true_label']}\n")
        else:
            print("Không có dự đoán sai nào!")
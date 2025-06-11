import pandas as pd

from src.data.utils import convert_label


class TrainDataLoader(object):
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, delimiter='\t', header=None, names=['label', 'content'], encoding="utf-8")
    
    def get_data(self):
        data = []
        for _, row in self.df.iterrows():
            data.append({
                    'category': convert_label(row['label']),
                    'content': row['content']
                }) 
        return data

class TestDataLoader:
    def __init__(self, content_path, label_path, encoding='utf-8'):
        """
        Khởi tạo đối tượng TestDataLoader.

        Args:
            content_path (str): Đường dẫn đến tệp chứa nội dung văn bản.
            label_path (str): Đường dẫn đến tệp chứa nhãn tương ứng.
            encoding (str, optional): Kiểu mã hóa của tệp (mặc định là 'utf-8').
        """
        try:
            with open(content_path, 'r', encoding=encoding) as f:
                self.content = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy tệp nội dung: {content_path}")

        try:
            with open(label_path, 'r', encoding=encoding) as f:
                self.labels = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy tệp nhãn: {label_path}")

        if len(self.content) != len(self.labels):
            raise ValueError("Số lượng nội dung và nhãn không khớp.")

    def get_data(self):
        """
        Trả về một danh sách các từ điển, mỗi từ điển chứa 'content' và 'label'.
        """
        data = []
        for content, label in zip(self.content, self.labels):
            data.append({
                'category': convert_label(label.strip()),
                'content': content.strip()
            })
        return data
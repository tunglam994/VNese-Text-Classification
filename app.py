from fastapi import FastAPI
from pydantic import BaseModel
from src.data.utils import *
from src.data.extract_features import *
from src.data.preprocess import *
from src.models.naive_bayes import Classifier
import numpy as np

app = FastAPI()

STOPWORDS_PATH = 'dataset/stopwords-nlp-vi.txt'
MODEL_PATH = 'src/models/ckpt/naive_model.pk'
DICTIONARY_PATH = 'dataset/dictionary.txt'

class InputText(BaseModel):
    input: str = None

@app.post('/')
async def predict(input_text: InputText):
    if input_text.input is None:
        return {"message": "No input text"}
    
    input = input_text.input
    
    # Preprocess input text
    preprocessor = Preprocessor(stopwords_path = STOPWORDS_PATH)
    features = preprocessor.get_words_feature(input)
    dense = np.array(get_dense(features, DICTIONARY_PATH)).reshape(1, -1)

    # Load classifier    
    classifier = Classifier()
    classifier.load_model(MODEL_PATH)

    # Predict
    label = classifier.model.predict(dense)[0]
    print(label)
    return {"label": label}

# if __name__ == "__main__":

#     input = 'Theo đó, mỗi cửa hàng bị xử phạt 35 triệu đồng, vì hành vi kinh doanh thực phẩm trong khi giấy chứng nhận cơ sở đủ điều kiện an toàn thực phẩm đã hết hiệu lực.'
    
#     # Preprocess input text
#     preprocessor = Preprocessor(stopwords_path = STOPWORDS_PATH)
#     features = preprocessor.get_words_feature(input)
#     dense = np.array(get_dense(features, DICTIONARY_PATH)).reshape(1, -1)

#     # Load classifier    
#     classifier = Classifier()
#     classifier.load_model(MODEL_PATH)

#     # Predict
#     label = classifier.model.predict(dense)[0]
#     print(label)
    



    
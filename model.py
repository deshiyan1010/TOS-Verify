import numpy as np
import tensorflow as tf
import sys
sys.path.append('models')
import tensorflow_hub as hub
from official.nlp.data import classifier_data_lib
from official.nlp.bert import tokenization

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# from official.nlp import optimization





class Model:
    def __init__(self):
        self.label_list = [0, 1] # Label categories
        self.max_seq_length = 128 # maximum length of (token) input sequences
        self.train_batch_size = 32
        self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",trainable=True)
        self.vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        self.do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(self.vocab_file, self.do_lower_case)


    def create_model(self):

        input_word_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                            name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                        name="input_mask")
        input_type_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                        name="input_type_ids")

        pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, input_type_ids])

        drop = tf.keras.layers.Dropout(0.4)(pooled_output)
        output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(drop)

        model = tf.keras.Model(
            inputs={
                'input_word_ids': input_word_ids,
                'input_mask': input_mask,
                'input_type_ids': input_type_ids
            },
            outputs=output)
        return model
    
    def to_feature_map(self,text, label):
        input_ids, input_mask, segment_ids, label_id = tf.py_function(self.to_feature, inp=[text, label], 
                                        Tout=[tf.int32, tf.int32, tf.int32, tf.int32])

        # py_func doesn't set the shape of the returned tensors.
        input_ids.set_shape([self.max_seq_length])
        input_mask.set_shape([self.max_seq_length])
        segment_ids.set_shape([self.max_seq_length])
        label_id.set_shape([])

        x = {
                'input_word_ids': input_ids,
                'input_mask': input_mask,
                'input_type_ids': segment_ids
            }
        return (x, label_id)


    def to_feature(self,text, label, ):
        label_list=self.label_list
        max_seq_length=self.max_seq_length
        tokenizer=self.tokenizer
        example = classifier_data_lib.InputExample(guid = None,
                                                    text_a = text.numpy(), 
                                                    text_b = None, 
                                                    label = label.numpy())
        feature = classifier_data_lib.convert_single_example(0, example, label_list,
                                            max_seq_length, tokenizer)
        
        return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)

    def load_model(self):
        self.model = self.create_model()
        self.model.load_weights('weights/weights')
         

    def predict(self,text_arr):
        test_data = tf.data.Dataset.from_tensor_slices((text_arr, [0]*len(text_arr)))
        test_data = (test_data.map(self.to_feature_map).batch(1))
        preds = self.model.predict(test_data)
        return preds


if __name__=="__main__":
    m = Model()
    m.load_model()
    print(m.predict(['If you are acting as a consumer, you agree to submit to the non-exclusive jurisdiction of the Irish courts.']))

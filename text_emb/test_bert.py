from tfbert import ALBertTokenizer, ALBertConfig, ALBertModel
import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == '__main__':

    config = ALBertConfig.from_pretrained("./data/albert_config.json")
    tokenizer = ALBertTokenizer.from_pretrained('./data/bert-base-cased-vocab.txt', do_lower_case=True)
    # model = BertModel(config, True, [1])
    batch_size = 2
    embedding_size = 100

    batch_sentences = [
        "But what about second breakfast?",
        "Don't think he knows about second breakfast, Pip.",
        "What about elevensies?",
    ]

    # encoded_dict = tokenizer.encode_plus(
    #     batch_sentences,  # Sentence to encode.
    #     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    #     max_length=64,  # Pad & truncate all sentences.
    #     return_attention_mask=True,  # Construct attn. masks.
    #     # return_tensors='pt',  # Return pytorch tensors.
    # )
    encoded_input = tokenizer(batch_sentences, padding=True)
    encoded_ids = [[1,2,3],[2,3,4]]

    input_ids = tf.Variable(encoded_ids)
    model = ALBertModel(config, True, input_ids)

    last_hidden_states_2 = model.outputs[0]     # pooler
    hidden_layer = tf.contrib.layers.fully_connected(last_hidden_states_2, 128,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(
                                                         seed=100))
    logits_flat = tf.contrib.layers.linear(hidden_layer, embedding_size,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(seed=102))
    logits = tf.reshape(logits_flat, [batch_size, embedding_size])
    print(last_hidden_states_2)
    print(logits)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # 执行初始化
        sess.run(init)

        # 打印结果
        print(last_hidden_states_2.eval())
        # print(b.eval())
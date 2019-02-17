import tensorflow as tf

from convNet_layers import conv_layer, fc_layer, flatten_tensor

def predict(x, dropout_rates, num_channels, num_labels):

    # Block 1
    filter_size1 = filter_size2 = 5
    num_filters1 = num_filters2 = 32

    # Block 2
    filter_size3 = filter_size4 = 5
    num_filters3 = num_filters4 = 64

    # Block 3
    filter_size5 = filter_size6 = filter_size7 = 5
    num_filters5 = num_filters6 = num_filters7 = 128

    # Fully-connected layers
    fc1_size = fc2_size = 256

    # Apply dropout to the input layer
    drop_input = tf.nn.dropout(x, dropout_rates[0])

    # Block 1
    conv_1 = conv_layer(drop_input, filter_size1, num_channels, num_filters1, "conv_1", pooling=False)
    conv_2 = conv_layer(conv_1, filter_size2, num_filters1, num_filters2, "conv_2", pooling=True)
    drop_block1 = tf.nn.dropout(conv_2, dropout_rates[1]) # Dropout

    # Block 2
    conv_3 = conv_layer(drop_block1, filter_size3, num_filters2, num_filters3, "conv_3", pooling=False)
    conv_4 = conv_layer(conv_3, filter_size4, num_filters3, num_filters4, "conv_4", pooling=True)
    drop_block2 = tf.nn.dropout(conv_4, dropout_rates[1]) # Dropout

    # Block 3
    conv_5 = conv_layer(drop_block2, filter_size5, num_filters4, num_filters5, "conv_5", pooling=False)
    conv_6 = conv_layer(conv_5, filter_size6, num_filters5, num_filters6, "conv_6", pooling=False)
    conv_7 = conv_layer(conv_6, filter_size7, num_filters6, num_filters7, "conv_7", pooling=True)
    flat_tensor, num_activations = flatten_tensor(tf.nn.dropout(conv_7, dropout_rates[2])) # Dropout

    # Fully-connected 1
    fc_1 = fc_layer(flat_tensor, num_activations, fc1_size, 'fc_1', relu=True)
    drop_fc2 = tf.nn.dropout(fc_1, dropout_rates[2]) # Dropout

    # Fully-connected 2
    fc_2 = fc_layer(drop_fc2, fc1_size, fc2_size, 'fc_2', relu=True)

    # Parallel softmax layers
    logits_1 = fc_layer(fc_2, fc2_size, num_labels, 'softmax1')
    logits_2 = fc_layer(fc_2, fc2_size, num_labels, 'softmax2')
    logits_3 = fc_layer(fc_2, fc2_size, num_labels, 'softmax3')
    logits_4 = fc_layer(fc_2, fc2_size, num_labels, 'softmax4')
    logits_5 = fc_layer(fc_2, fc2_size, num_labels, 'softmax5')

    y_pred = tf.stack([logits_1, logits_2, logits_3, logits_4, logits_5])

    # The class-number is the index of the largest element
    y_pred_cls = tf.transpose(tf.argmax(y_pred, dimension=2))

    return y_pred, y_pred_cls

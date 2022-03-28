import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from DeepConvLSTM import get_DCL
from  Augment import *
import tensorflow_addons as tfa
import time
def attch_projection_head(backbone,dim1=256,dim2=128,dim3=50):
    return Sequential([backbone,Dense(dim1),ReLU(),Dense(dim2),ReLU(),Dense(dim3)])

def contrastive_loss(out, out_aug, batch_size, hidden_norm=True, temperature=1.0, weights=1.0):
    LARGE_NUM = 1e9
    entropy_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    h1 = out
    h2 = out_aug
    if hidden_norm:
        h1 = tf.math.l2_normalize(h1, axis=1)
        h2 = tf.math.l2_normalize(h2, axis=1)

    labels = tf.range(batch_size)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(h1, h1, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(h2, h2, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM

    logits_ab = tf.matmul(h1, h2, transpose_b=True) / temperature
    logits_ba = tf.matmul(h2, h1, transpose_b=True) / temperature

    loss_a = entropy_function(labels, tf.concat([logits_ab, logits_aa], 1), sample_weight=weights)
    loss_b = entropy_function(labels, tf.concat([logits_ba, logits_bb], 1), sample_weight=weights)
    loss = loss_a + loss_b
    return loss

def train_step(xis, xjs, model, optimizer, temperature):
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)
        batch_size = len(xis)
        loss = contrastive_loss(zis, zjs, batch_size, hidden_norm=True, temperature=temperature)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def evaluate(model_cl,reserve_layer,outputs,method='linear'):
    if method=='linear':
        model = Model(model_cl.layers[0].input,model_cl.layers[reserve_layer].output,trainable=False)
    else:
        model = Model(model_cl.layers[0].input, model_cl.layers[reserve_layer].output, trainable=True)
    return Sequential([model,Dense(outputs,activation='softmax')])


def main():
    x_data = np.load('datasets/UCI_X.npy')
    y_data = np.load('datasets/UCI_Y.npy')

    n_timesteps, n_features, n_outputs = x_data.shape[1], x_data.shape[2], y_data.shape[1]
    backbone = get_DCL(n_timesteps, n_features)
    model_cl = attch_projection_head(backbone)

    batch_size = 1024
    epochs = 200
    temperature = 0.1
    optimizer = tf.keras.optimizers.Adam(0.001)

    for epoch in range(epochs):
        loss_epoch = []
        train_loss_dataset = tf.data.Dataset.from_tensor_slices(x_data).shuffle(len(y_data),reshuffle_each_iteration=True).batch(batch_size)
        for x in train_loss_dataset:
            xis = resampling_fast_random(x)  #Select the augmentation method used
            xjs = noise(x)              #Select the augmentation method used
            loss = train_step(xis, xjs, model_cl, optimizer, temperature=temperature)
            loss_epoch.append(loss)
        print("epoch{}===>loss:{}".format(epoch + 1, np.mean(loss_epoch)))
    timestamp =  time.time()
    model_cl.save('contrastive_model/SimCLRHAR_'+str(timestamp)+'.h5')

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.99)

    linear_model = evaluate(model_cl,-6,n_outputs,'linear')
    linear_model.compile(loss="categorical_crossentropy", metrics=[tfa.metrics.F1Score(num_classes=n_outputs,average='micro')],
                         optimizer=tf.keras.optimizers.Adam(0.01))
    history_linear = linear_model.fit(x_train, y_train, epochs=200, batch_size=50, validation_data=(x_test, y_test), shuffle=True)
    print('linear best accuracy: {}'.format(np.max(history_linear.history['val_f1_score'])))

    fine_model = evaluate(model_cl, -6, n_outputs, 'fine')
    fine_model.compile(loss="categorical_crossentropy", metrics=[tfa.metrics.F1Score(num_classes=n_outputs,average='micro')],
                         optimizer=tf.keras.optimizers.Adam(0.0005))
    history_fine = fine_model.fit(x_train, y_train, epochs=200, batch_size=50, validation_data=(x_test, y_test),
                                      shuffle=True)
    print('fine best accuracy: {}'.format(np.max(history_fine.history['val_f1_score'])))


if __name__ == '__main__':
    main()


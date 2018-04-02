# -*- coding: utf-8 -*-

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

K.set_image_data_format('channels_last')

def CapsNet(input_shape, n_class, num_routing):

    x = layers.Input(shape=input_shape )

    conv1 = layers.Conv2D(filters=256, kernel_size=7, strides=1, padding='valid', activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, num_routing=num_routing,
                             name='digitcaps')(primarycaps)

    out_caps = Length(name='capsnet')(digitcaps)

    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])
    masked = Mask()(digitcaps)

    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    return train_model, eval_model


def margin_loss(y_true, y_pred):

    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def train(model, data, args):

    (x_train, y_train), (x_test, y_test) = data

    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/best-weight.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})


    def train_generator(x, y, batch_size):
        train_datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)

        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model

def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=32)
    print('-'*50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
    print()
    print('Reconstructed images are saved to real_and_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("real_and_recon.png", ))
    plt.show()

def veri_yukle():

    import numpy as np
    veriseti_klasoru = 'dataset'
    x_train = np.load(veriseti_klasoru+'/x_train.npy')
    x_test = np.load(veriseti_klasoru+'/x_test.npy')
    y_train = np.load(veriseti_klasoru+'/y_train.npy')
    y_test = np.load(veriseti_klasoru+'/y_test.npy')

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks
    from keras.utils.vis_utils import plot_model

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    #parser.add_argument('--lam_recon', default=0.392, type=float)  # 784 * 0.0005
    parser.add_argument('--lam_recon', default=0.512, type=float)  # 784 * 0.0005
    parser.add_argument('--num_routing', default=3, type=int)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 TensorBoard’ta ağırlıklar tutulur.
    parser.add_argument('--save_dir', default='result')
    parser.add_argument('--is_training', default=0, type=int)
    parser.add_argument('--weights', default='pretrained_model_siu2018_capsulenet.h5')
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    (x_train, y_train), (x_test, y_test) = veri_yukle()

    model, eval_model = CapsNet(input_shape=x_train.shape[1:],
                                n_class=len(np.unique(np.argmax(y_train, 1))),
                                num_routing=args.num_routing)
    model.summary()
    plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)

    if args.weights is not None:  # Model ağırlıkları verilir
        model.load_weights(args.weights)
    if args.is_training:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # Ağırlıkları verildiği sürece test işlemi yapılır.
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=eval_model, data=(x_test, y_test))

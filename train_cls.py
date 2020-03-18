import os
import json
import pandas as pd

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

def generate(batch, shape, ptrain, pval):
    """Data generation and augmentation

    # Arguments
        batch: Integer, batch size.
        size: Integer, image size.
        ptrain: train dir.
        pval: eval dir.

    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        count1: Integer, number of train set.
        count2: Integer, number of test set.
    """

    #  Using the data Augmentation in traning data
    datagen1 = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen2 = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen1.flow_from_directory(
        ptrain,
        target_size=shape,
        batch_size=batch,
        class_mode='categorical')

    validation_generator = datagen2.flow_from_directory(
        pval,
        target_size=shape,
        batch_size=batch,
        class_mode='categorical')

    count1 = 0
    for root, dirs, files in os.walk(ptrain):
        for each in files:
            count1 += 1

    count2 = 0
    for root, dirs, files in os.walk(pval):
        for each in files:
            count2 += 1

    return train_generator, validation_generator, count1, count2


def train():
    with open('config/config.json', 'r') as f:
        cfg = json.load(f)

    save_dir = cfg['save_dir']
    shape = (int(cfg['height']), int(cfg['width']), 3)
    n_class = int(cfg['class_number'])
    batch = int(cfg['batch'])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if cfg['model'] == 'large':
        from model.mobilenet_v3_large import MobileNetV3_Large
        model = MobileNetV3_Large(shape, n_class).build()
    if cfg['model'] == 'small':
        from model.mobilenet_v3_small import MobileNetV3_Small
        model = MobileNetV3_Small(shape, n_class).build()

    pre_weights = cfg['weights']
    if pre_weights and os.path.exists(pre_weights):
        model.load_weights(pre_weights, by_name=True)

    opt = Adam(lr=float(cfg['learning_rate']))
    earlystop = EarlyStopping(monitor='val_acc', patience=5, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_dir, '{}_weights.h5'.format(cfg['model'])),
                 monitor='val_acc', save_best_only=True, save_weights_only=True)
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    train_generator, validation_generator, count1, count2 = generate(batch, shape[:2], cfg['train_dir'], cfg['eval_dir'])

    hist = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=count1 // batch,
        validation_steps=count2 // batch,
        epochs=cfg['epochs'],
        callbacks=[earlystop,checkpoint])

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(os.path.join(save_dir, 'hist.csv'), encoding='utf-8', index=False)
    #model.save_weights(os.path.join(save_dir, '{}_weights.h5'.format(cfg['model'])))


if __name__ == '__main__':
    train()

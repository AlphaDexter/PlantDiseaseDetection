from CreateModel import createModel
from LoadImage import loadimage
import numpy as np
from keras import callbacks

model1 = createModel()
X_train, X_test, y_train, y_test, input_shape = loadimage()
filename = 'model_train_new.csv'


def training(X_train, X_test, y_train, y_test, input_shape):

    num_epoch = 20

    model1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

    model1.summary()
    model1.get_config()
    model1.layers[0].get_config()
    model1.layers[0].input_shape()
    model1.layers[0].output_shape()
    model1.layers[0].get_weights()
    np.shape(model1.layers[0].get_weights()[0])
    model1.layers[0].trainable()

    model1.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))

    csv_log = callbacks.CSVLogger(filename, separator=',', append=False)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

    filepath = "Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    callbacks_list = [csv_log, early_stopping, checkpoint]

    model1.fit(X_train, y_train, batch_size=16, nb_epoch=5, verbose=1, validation_data=(X_test, y_test), callbacks=callbacks_list)

    model1.save('Plant_Disease_model.h5')

    score = model1.evaluate(X_test, y_test, verbose=0)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    test_image = X_test[0:1]
    print (test_image.shape)

    print(model1.predict(test_image))
    print(model1.predict_classes(test_image))
    print(y_test[0:1])

    return True

from keras import backend as K
from keras.models import model_from_json
from keras import losses, metrics, optimizers


class Model:
    def __init__(self):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights("model.h5")
        self.loaded_model.compile(optimizer=optimizers.RMSprop(lr = 0.01),
                                  loss=losses.binary_crossentropy,
                                  metrics=[metrics.binary_accuracy])

    def processSingle(self, Interface, bands):
        mask = self.networkPredict(bands)[0]
        for i in range(16):
            Interface.progress()
        K.clear_session()
        return mask

    def networkPredict(self, bands):
        return self.loaded_model.predict(bands.reshape((-1, 256, 256, bands.shape[2])))
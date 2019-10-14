from keras.models import load_model
class l_model:

    def __init__(self):

        self.model_address = 'a/model_dir/chest.h5'
        self.model = load_model(self.model_address)
        self.model._make_predict_function()
        # model.summary()



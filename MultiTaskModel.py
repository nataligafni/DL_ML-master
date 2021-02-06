import os
from model import network_model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import datetime
from time import strftime
from paths_and_params.Configuration import Configuration
from paths_and_params.params import Params

config = Configuration()
params = Params()


class MultiTaskModel():
    def __init__(self,epochs,training,validation, test, phase):
        self.save_path = config.save_model
        self.epochs = epochs
        self.training_set = training
        self.validation_set = validation
        self.test_set = test
        self.phase=phase
        if phase=='train':
            self.run_model=self.run_model()


    def run_model(self):

        model = network_model()
        if self.phase == 'train':

            os.mkdir(config.run_dir)
            callbacks = [
                         # EarlyStopping(patience=15, verbose=1),
                         ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.000001, verbose=1),
                         ModelCheckpoint(config.run_dir + '/weights_e{epoch:04d}_loss{loss:.4f}_val_loss{val_loss:.4f}.h5', verbose=1,
                                         save_best_only=False, save_weights_only=True)]
            try:
                all_subdirs = [d for d in os.listdir(config.save_model) if os.path.isdir(config.save_model)]
                full_subdirs = [(config.save_model + '\\' + d) for d in all_subdirs]
                latest_subdir = max(full_subdirs, key=os.path.getmtime)
                latest_subdir = latest_subdir + '/MTL_weights_final_epoch.h5'
                # latest_subdir = latest_subdir + '/weights_e0007_loss0.5930_val_loss0.7217.h5'
                # Loads the weights
                # model.load_weights(latest_subdir)
                model.load_weights(config.weights_path)
                print('Loaded the following weights: ' + config.weights_path)

            except:
                print('WARNING: did not load weights!!!!!!!!!')
            output = model.fit_generator(generator=self.training_set,
                                         validation_data=self.validation_set, use_multiprocessing=False,
                                         workers=1, epochs=self.epochs, callbacks=callbacks, verbose=1)

            ### save model

            model.save_weights(config.run_dir + '/MTL_weights_final_epoch.h5')

        else:

            all_subdirs = [d for d in os.listdir(config.save_model) if os.path.isdir(config.save_model)]
            full_subdirs= [(config.save_model + '\\' +d) for d in all_subdirs ]
            latest_subdir = max(full_subdirs, key=os.path.getmtime)
            latest_subdir = latest_subdir + '/MTL_weights_final_epoch.h5'
            # Loads the weights
            # model.load_weights(latest_subdir)
            model.load_weights(config.weights_path)
            print('Loaded the following weights: ' + config.weights_path)

            output=model.predict(self.test_set, verbose=1)

            # I_predict_lab_all, = np.argmax(I_predict_score_all, axis=3)

        return output



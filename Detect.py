import tensorflow as tf
import keras
from keras.models import load_model



def main():
        
    model = load_model('stamp_recog.h5')
    test_path = 'Images/Testing'
    test_batches = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path, target_size = (224,224), batch_size = 3, shuffle = False)
    predictions = model.predict( x = test_batches , verbose = 0).argmax(axis=-1)
    result = ""
    ReturnResult = ""
    counter = 0
    x = len(predictions)
    for i in range (0,x):
        if predictions[i] == 0:
            result = "Non Stamp image"
        else:
            result = "Stamp image"
        ReturnResult = ReturnResult + test_batches.filenames[counter] + " - "+ result + "\n"
        counter += 1


    return ReturnResult


main()

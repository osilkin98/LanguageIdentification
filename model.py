import keras as ks
import load_data as ld
from sklearn.model_selection import train_test_split
import json

# Create the model given a number of languages
def create_model(num_languages):

    features, embed_dimension = ld.max_features, 128
    model = ks.Sequential()
    # Outputs a matrix of size 100x128 with 1,280,000 parameters (10000x100x128=1280000)
    model.add(ks.layers.Embedding(input_dim=features,  # Input dimension
                                  output_dim=embed_dimension,  # Dimension to output
                                  input_length= 100))  # Length of the input sequence

    # Outputs a matrix of size 100x128 with 49280 parameters to train
    model.add(ks.layers.Conv1D(filters=embed_dimension,  # Convolve using 128 filters
                               kernel_size=5,            # with a kernel size of 3
                               padding='same',           # Use 0 padding
                               dilation_rate=1,          # We'll using D = 1 so that way the kernels are densely packed
                               activation='relu'))       # ReLU activation layer since it works well

    # Reduces the shape of the output matrix from (100x128) to (100/4x128) => (25x128)
    # We'll use a pool size of 4 at first although we may want to change this
    model.add(ks.layers.MaxPooling1D(pool_size=4))

    # Next we'll reduce the amount of filters being used by 1/2 so that the output is (25x64)
    model.add(ks.layers.Conv1D(filters=64,  # 128/2
                               kernel_size=3,            # The same kernel size as last time
                               padding='same',           # 0 padding as last time
                               dilation_rate=1,          # Same dilation rate of 1
                               activation='relu'))       # Relu activation again

    # Now we'll reduce the output size from (25x64) to (25/2x64) => (floor(12.5)x64) => (12x64)
    # Using a max pooling layer with a pool size of just 2
    model.add(ks.layers.MaxPooling1D(pool_size=2))

    # Now we'll add a bi-directional LSTM with an output size of (64)
    model.add(ks.layers.CuDNNLSTM(units=64))

    # Now we'll add a dropout layer with a dropout rate of 30%, output is still (64)
    model.add(ks.layers.Dropout(rate=0.5))

    # this will feed into a dense layer with an input of (64) and an output of (64)
    model.add(ks.layers.Dense(units=64,
                              activation='sigmoid'  # going to use sigmoid as an activation function
                              ))

    # which will in turn feed into a final dense layer with an output of (num_languages)
    model.add(ks.layers.Dense(units=num_languages,  # Since this is a many-to-many classification problem,
                              activation='softmax'  # the softmax activation function will work well
                              ))

    model.compile(loss='categorical_crossentropy',  # We use the categorical crossentropy function as a loss metric
                  optimizer='adam',                 # because this is a many-to-many classification problem
                  metrics= ['accuracy'])

    return model


# Main function to perform the training and testing operations with
def main_train():
    # Get the data, [X -> code samples, Y -> language labels]
    code, languages = ld.create_tokenized_data_set(json_file="data_index.json", overwrite_file=True)

    training_code, testing_code, training_labels, testing_labels = train_test_split(code, languages,
                                                                                    test_size=0.2, random_state=42)

    print("Training_code shape: {}\nTesting_code shape: {}".format(training_code.shape, testing_code.shape))

    model = create_model(len(languages.columns))

    print("Model Summary: {}".format(model.summary()))
    print("Languages: {}".format(languages.columns))

    batch_size = 32

    history = model.fit(code, languages, epochs=400, batch_size=batch_size)

    model.save(filepath="saved_models/code_model.h5")
    model.save_weights(filepath="saved_models/code_model_weights.h5")
    score, accuracy = model.evaluate(testing_code, testing_labels, verbose=2, batch_size=batch_size)

    print("Metric names: {}\nValidation loss: {}\nValidation Accuracy: {}".format(model.metrics_names,
                                                                                  score,
                                                                                  accuracy))


def evaluate_file(filename):
    model = ks.models.load_model(filepath="saved_models/code_model.h5")
    model.load_weights(filepath="saved_models/code_model_weights.h5")
    to_eval = ld.create_input_evaluation(filename)
    to_eval = to_eval[0].reshape(1, to_eval.shape[1])
    print("To Eval: {}".format(to_eval))
    language_probabilities = model.predict(to_eval, batch_size=1, verbose=2)[0]
    print("Language Probabilities: {}".format(language_probabilities))
    data = {"success": False}
    data["predictions"] = []

    for i in range(len(ld.languages)):
        r = {"label": ld.languages[i], "probability": "{0:.3f}% ".format(language_probabilities[i]*100)}
        data["predictions"].append(r)
    data["success"] = True

    print(json.dumps(data, indent=4))


if __name__ == "__main__":
    evaluate_file("test.cpp")
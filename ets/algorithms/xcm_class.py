from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    Reshape,
    Conv1D,
    concatenate,
    GlobalAveragePooling1D,
    Dense,
)

class XCMModel:
    def __init__(self, input_shape, n_class, window_size, filters_num=128):
        """
        Initializes the XCM model parameters.

        Parameters
        ----------
        input_shape: tuple
            Shape of the input (timesteps, features, 1).

        n_class: int
            Number of classes for classification.

        window_size: float
            Size of the subsequence of the MTS expected to be interesting.

        filters_num: int, optional
            Number of filters in convolution layers. Default is 128.
        """
        self.input_shape = input_shape
        self.n_class = n_class
        self.window_size = window_size
        self.filters_num = filters_num
        self.model = None

    def build_model(self):
        """
        Builds the XCM model architecture.

        Returns
        -------
        model : Model
            Compiled Keras Model instance.
        """
        # Input layer shape
        n = self.input_shape[0]
        k = self.input_shape[1]
        input_layer = Input(shape=(n, k, 1))

        # 2D convolution layers
        a = Conv2D(
            filters=int(self.filters_num),
            kernel_size=(int(self.window_size * n), 1),
            strides=(1, 1),
            padding="same",
            input_shape=(n, k, 1),
            name="2D",
        )(input_layer)
        a = BatchNormalization()(a)
        a = Activation("relu", name="2D_Activation")(a)
        a = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), name="2D_Reduced")(a)
        a = Activation("relu", name="2D_Reduced_Activation")(a)
        x = Reshape((n, k))(a)

        # 1D convolution layers
        b = Reshape((n, k))(input_layer)
        b = Conv1D(
            filters=int(self.filters_num),
            kernel_size=int(self.window_size * n),
            strides=1,
            padding="same",
            name="1D",
        )(b)
        b = BatchNormalization()(b)
        b = Activation("relu", name="1D_Activation")(b)
        b = Conv1D(filters=1, kernel_size=1, strides=1, name="1D_Reduced")(b)
        y = Activation("relu", name="1D_Reduced_Activation")(b)

        # Concatenation
        z = concatenate([x, y])

        # 1D convolution layer
        z = Conv1D(
            filters=self.filters_num,
            kernel_size=int(self.window_size * n),
            strides=1,
            padding="same",
            name="1D_Final",
        )(z)
        z = BatchNormalization()(z)
        z = Activation("relu", name="1D_Final_Activation")(z)

        # 1D global average pooling and classification
        z = GlobalAveragePooling1D()(z)
        output_layer = Dense(self.n_class, activation="softmax")(z)

        # Creating the model
        self.model = Model(input_layer, output_layer)
        print("XCM Model Built")
        return self.model

    def summary(self):
        """
        Prints the model summary.
        """
        if self.model is None:
            raise ValueError("Model is not built yet. Call `build_model` first.")
        return self.model.summary()

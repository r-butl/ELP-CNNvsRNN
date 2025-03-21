import tensorflow as tf
from cnn_config import CNNConfig

class CNN(tf.keras.Model):
    def __init__(self, model_config, training, input_shape=None):
        super(CNN, self).__init__()

        self.cfg = model_config
        self.training = training

        if self.cfg['activation_function'] == 'ReLU':
            self.activation_function = tf.nn.relu
        elif self.cfg['activation_function'] == 'LeakyReLU':
            activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.2)
            self.activation_function = activation_fn

        # If input_shape is None, create a dynamic input placeholder
        if input_shape:
            self.input_layer = tf.keras.layers.Input(shape=input_shape)
        else:
            self.input_layer = None  # Will be inferred later

        # Load ResNet50 with dynamic input handling
        self.resnet = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,  # Train from scratch
            input_shape=input_shape if input_shape else (None, None, 1)  # Allow dynamic size
        )

        if self.training: 
            self.resnet.trainable = True  # Ensure all layers are trainable
        else:
            self.resnet.trainable = False

        # Flatten the output
        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layers
        fc_init = tf.keras.initializers.GlorotNormal(seed=42)
        self.drop1 = tf.keras.layers.Dropout(self.cfg['dropout_rate'])
        self.fc1 = tf.keras.layers.Dense(256, activation=self.activation_function, kernel_initializer=fc_init)
        self.fc2 = tf.keras.layers.Dense(50, activation=self.activation_function, kernel_initializer=fc_init)
        self.out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=fc_init)

    def call(self, x):
        x = self.resnet(x, training=self.training)
        x = self.flatten(x)
        x = self.fc1(x)
        if self.training:
            x = self.drop1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

if __name__ == "__main__":

    config = {  
            "learning_rate": 0.0001,
            "learning_rate_decay_steps": 500,
            "learning_rate_decay": 0.97,
            "momentum": 0.9,
            "batch_size": 8,
            "epochs": 300,
            "activation_function": "LeakyReLU",
            "dropout_rate": 0.2,
            "optimizer": "sgd",
            "config": CNNConfig,
            "patience": 10,
            "min_delta": 0.001,
        }
    # Ensure the model is built
    model = CNN(model_config=config, training=True, input_shape=(563, 98, 1))  # Example input shape

    # If needed, you can call build() explicitly:
    model.build(input_shape=(None, 563, 98, 1))  # None allows for batch size flexibility

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Now you can safely call summary
    model.summary()

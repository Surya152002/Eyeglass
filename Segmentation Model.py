import tensorflow as tf

class SegmentationModel:
    def __init__(self):
        self.model = None

    def load_model(self, weights_path):
        self.model = tf.keras.models.load_model(weights_path)

    def segment_image(self, image):
        segmented_image = self.model.predict(np.expand_dims(image, axis=0))
        return segmented_image

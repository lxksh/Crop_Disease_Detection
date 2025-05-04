import tensorflow as tf

# Load your old model (may need to do this on a machine where it works)
model = tf.keras.models.load_model("models/saved_models/best_model.h5", compile=False)

# Save it again with the new TensorFlow version
model.save("models/saved_models/best_model_219.h5")
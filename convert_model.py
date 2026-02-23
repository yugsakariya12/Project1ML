import tensorflow as tf
from keras.engine.input_layer import InputLayer

# 🔥 Monkey-patch InputLayer to accept batch_shape
_original_init = InputLayer.__init__

def _patched_init(self, *args, **kwargs):
    if "batch_shape" in kwargs:
        kwargs["input_shape"] = kwargs["batch_shape"][1:]
        kwargs.pop("batch_shape")
    _original_init(self, *args, **kwargs)

InputLayer.__init__ = _patched_init


# ✅ Now load & convert the model
model = tf.keras.models.load_model("models/phishing_model.h5", compile=False)

model.save("models/phishing_model_fixed.keras")

print("✅ Model converted successfully")
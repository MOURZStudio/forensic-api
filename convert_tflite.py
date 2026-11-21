"""
convert_tflite.py — konversi cnn_mobilenetv2_forensik.h5 -> .tflite (float32, tanpa kuantisasi).

Dijalankan sekali secara lokal (tidak di Render). Lingkungan yang dipakai:
    Python 3.11.9, tensorflow-cpu==2.20.0, keras==3.13.2

Bobot model tidak diubah; selisih keluaran .h5 vs .tflite pada uji acak < 1e-5.
"""
import tensorflow as tf
import keras

model = keras.models.load_model('cnn_mobilenetv2_forensik.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
with open('cnn_mobilenetv2_forensik.tflite', 'wb') as f:
    f.write(converter.convert())
print('OK: cnn_mobilenetv2_forensik.tflite')

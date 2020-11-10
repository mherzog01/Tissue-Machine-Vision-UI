import tensorflow as tf
import numpy as np

a = np.random.random((2,5,4))
print(f'a={a}')

mult_div = 10**4

t = tf.convert_to_tensor(a)
det_val = tf.math.round(t * mult_div) / mult_div
#det_val = tf.cast(det_val, tf.float16) 
print('No cast')
print(det_val.numpy().tolist())
del t, det_val

t2 = tf.convert_to_tensor(a)
det_val2 = tf.math.round(t2 * mult_div) / mult_div
det_val2 = tf.cast(det_val2, tf.float16) 
print('Cast')
print(det_val2.numpy().tolist())
del t2, det_val2

t3 = tf.convert_to_tensor(a)
det_val3 = (tf.math.floor(t3 * mult_div) + 0.5) / mult_div
det_val3 = tf.cast(det_val3, tf.float16) 
print('Cast - Floor')
print(det_val3.numpy().tolist())
del t3, det_val3

a4 = a.copy()
det_val4 = np.round(a4 * mult_div) / mult_div
#det_val4 = det_val4.astype(np.float16) 
print('No cast - numpy')
print(det_val4.tolist())
del a4, det_val4

a5 = a.copy()
det_val5 = np.round(a5 * mult_div) / mult_div
det_val5 = det_val5.astype(np.float16) 
print('Cast - numpy')
print(det_val5.tolist())


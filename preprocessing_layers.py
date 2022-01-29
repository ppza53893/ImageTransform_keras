import math
from typing import Union

import tensorflow as tf
from tensorflow.keras import Sequential, backend, layers

if tf.__version__ > '2.6.0':
    from keras.layers.preprocessing.image_preprocessing import transform
    from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
else:
    # direct
    from tensorflow.keras.layers.experimental.preprocessing import (
        RandomFlip, RandomRotation, RandomZoom)
    from tensorflow.python.keras.layers.preprocessing.image_preprocessing import \
        transform

SEED = 0
rng = tf.random.Generator.from_seed(SEED)


def smart_cond(pred, true_fn=None, false_fn=None, name=None):
  if isinstance(pred, tf.Variable):
    return tf.cond(
        pred, true_fn=true_fn, false_fn=false_fn, name=name)
  return tf.__internal__.smart_cond.smart_cond(
      pred, true_fn=true_fn, false_fn=false_fn, name=name)


class RandomBrightness(layers.Layer):
    def __init__(self,
                 brightness: Union[float, tuple] = 0.2,
                 seed: int = SEED,
                 **kwargs):
        super(RandomBrightness, self).__init__(**kwargs)
        if isinstance(brightness, (list, tuple)):
            assert brightness[0] < brightness[1] and brightness[1] >= 0.0, 'brightness must be non-negative'
            self.brightness = brightness
        else:
            assert brightness >= 0, 'brightness must be non-negative'
            self.brightness = (-brightness, brightness)
        if seed != SEED:
            self.rng = tf.random.Generator.from_seed(seed)
        else:
            self.rng = rng

    def call(self, inputs, training=None):
        if training is None:
            training = backend.learning_phase()
        original_shape = inputs.shape
        
        def random_channel_shift():
            delta = self.rng.uniform(
                (), self.brightness[0], self.brightness[1], dtype=inputs.dtype)
            return tf.image.adjust_brightness(inputs, delta)

        outputs = smart_cond(
            training, true_fn=random_channel_shift, false_fn=lambda: inputs)
        outputs.set_shape(original_shape)
        return outputs

    def get_config(self):
        config = {
            'brightness': self.brightness,
            'seed': self.rng.seed
        }
        base_config = super(RandomBrightness, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomChannelShift(layers.Layer):
    def __init__(self,
                 shift_range: float,
                 seed: int = SEED,
                 **kwargs):
        super(RandomChannelShift, self).__init__(**kwargs)
        self.shift_range = shift_range
        self.seed = seed
        if seed != SEED:
            self.rng = tf.random.Generator.from_seed(seed)
        else:
            self.rng = rng

    def call(self, inputs, training=None):
        if training is None:
            training = backend.learning_phase()
        original_shape = inputs.shape
        
        def random_channel_shift():
            cmin = tf.reduce_min(
                inputs, axis=[1, 2], keepdims=True)
            cmax = tf.reduce_max(
                inputs, axis=[1, 2], keepdims=True)
            avalue = self.rng.uniform((), -self.shift_range, self.shift_range, dtype=inputs.dtype)
            clipped = tf.clip_by_value(inputs + avalue, cmin, cmax)
            return clipped

        outputs = smart_cond(
            training, true_fn=random_channel_shift, false_fn=lambda: inputs)
        outputs.set_shape(original_shape)
        return outputs
    
    def get_config(self):
        config = {
            'shift_range': self.shift_range,
            'seed': self.seed,
        }
        base_config = super(RandomChannelShift, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomShift(layers.Layer):
    def __init__(self,
                 width_shift_range: float,
                 height_shift_range: float,
                 fill_mode: str = 'nearest',
                 fill_value: Union[float, int] = 0.,
                 seed: int = SEED,
                 **kwargs):
        
        super(RandomShift, self).__init__(**kwargs)
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed
        if seed != SEED:
            self.rng = tf.random.Generator.from_seed(seed)
        else:
            self.rng = rng
    
    
    def call(self, inputs, training=None):
        if training is None:
            training = backend.learning_phase()
        original_shape = inputs.shape
        
        def random_shift():
            input_shape = tf.shape(inputs)
            h, w = input_shape[1], input_shape[2]
            h = tf.cast(h, tf.float32)
            w = tf.cast(w, tf.float32)
            tx = self.rng.uniform((), -self.height_shift_range, self.height_shift_range)*h
            ty = self.rng.uniform((), -self.width_shift_range, self.width_shift_range)*w

            shift_matrix = tf.convert_to_tensor(
                [[1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]])
            transform_matrix = shift_matrix @ shift_matrix
            transform_matrix = tf.reshape(transform_matrix, [-1])[:8]
            transform_matrix = tf.tile(tf.expand_dims(transform_matrix, 0), [tf.shape(inputs)[0], 1])
            return transform(inputs, transform_matrix, fill_mode=self.fill_mode, fill_value=self.fill_value)
        
        outputs = smart_cond(
            training, true_fn=random_shift, false_fn=lambda: inputs)
        outputs.set_shape(original_shape)
        return outputs

    def get_config(self):
        config = {
            'width_shift_range': self.width_shift_range,
            'height_shift_range': self.height_shift_range,
            'fill_mode': self.fill_mode,
            'fill_value': self.fill_value,
            'seed': self.seed,
        }
        base_config = super(RandomShift, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomShear(layers.Layer):
    def __init__(
        self,
        shear_range: float,
        fill_mode: str = 'nearest',
        fill_value: Union[float, int] = 0.,
        seed: int = SEED,
        **kwargs):
        super(RandomShear, self).__init__(**kwargs)
        self.shear_range = shear_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed
        if seed != SEED:
            self.rng = tf.random.Generator.from_seed(seed)
        else:
            self.rng = rng

    def call(self, inputs, training=None):
        if training is None:
            training = backend.learning_phase()
        original_shape = inputs.shape
        
        def random_shear():
            shear = self.shear_range / 180 * math.pi
            shear = self.rng .uniform((), -self.shear_range, self.shear_range)
            shear_matrix = tf.convert_to_tensor(
                [[1, -tf.sin(shear), 0],
                [0, tf.cos(shear), 0],
                [0, 0, 1]])
            transform_matrix = shear_matrix @ shear_matrix
            transform_matrix = tf.reshape(transform_matrix, [-1])[:8]
            transform_matrix = tf.tile(tf.expand_dims(transform_matrix, 0), [tf.shape(inputs)[0], 1])
            return transform(inputs, transform_matrix, fill_mode=self.fill_mode, fill_value=self.fill_value)
        
        outputs = smart_cond(training, true_fn=random_shear, false_fn=lambda: inputs)
        outputs.set_shape(original_shape)
        return outputs

    def get_config(self):
        config = {
            'shear_range': self.shear_range,
            'fill_mode': self.fill_mode,
            'fill_value': self.fill_value,
            'seed': self.seed,
        }
        base_config = super(RandomShear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ImageTransformation(layers.Layer):
    def __init__(self,
                 rotation_range: Union[float, tuple, list]=0,
                 width_shift_range: Union[float, tuple, list]=0.,
                 height_shift_range: Union[float, tuple, list]=0.,
                 brightness_range: Union[float, tuple]=None,
                 shear_range: float=0.,
                 zoom_range: Union[float, tuple, list]=0.,
                 channel_shift_range: Union[float, tuple, list]=0.,
                 fill_mode: str='nearest',
                 horizontal_flip: bool=False,
                 vertical_flip: bool=False,
                 fill_value: Union[float, int] = None,
                 seed: int = SEED,
                 **kwargs):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.shear_range: float = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode or 'nearest'
        self.flip_mode = ''
        if horizontal_flip:
            self.flip_mode += 'horizontal'
        if vertical_flip:
            if self.flip_mode:
                self.flip_mode += '_and_'
            self.flip_mode += 'vertical'
        if fill_value is None and 'cval' in kwargs:
            self.fill_value = kwargs.pop('cval')
        else:
            self.fill_value = fill_value or 0.
        self.seed = seed
        super(ImageTransformation, self).__init__(**kwargs)


    def build(self, input_shape):
        layers = [
            RandomRotation(
                self.rotation_range, fill_mode=self.fill_mode, fill_value=self.fill_value, seed=self.seed),
            RandomShift(
                self.width_shift_range, self.height_shift_range, self.fill_mode, fill_value=self.fill_value, seed=self.seed),
            RandomShear(
                self.shear_range, self.fill_mode, fill_value=self.fill_value, seed=self.seed),
            RandomZoom(
                self.zoom_range, fill_mode=self.fill_mode, fill_value=self.fill_value, seed=self.seed),
            ]
        if self.flip_mode != '':
            layers.append(RandomFlip(self.flip_mode, seed=self.seed))
        layers.append(RandomChannelShift(self.channel_shift_range, seed=self.seed))
        if self.brightness_range is not None:
            layers.append(RandomBrightness(self.brightness_range, seed=self.seed))
        self.preprocessing = Sequential(layers=layers)

    def call(self, inputs, training=None):
        if training is None:
            training = backend.learning_phase()
        return self.preprocessing(inputs, training=training)

    def get_config(self):
        config = {
            'rotation_range': self.rotation_range,
            'width_shift_range': self.width_shift_range,
            'height_shift_range': self.height_shift_range,
            'brightness_range': self.brightness_range,
            'shear_range': self.shear_range,
            'zoom_range': self.zoom_range,
            'channel_shift_range': self.channel_shift_range,
            'fill_mode': self.fill_mode,
            'horizontal_flip': self.flip_mode.startswith('horizontal'),
            'vertical_flip': self.flip_mode.endswith('vertical'),
            'fill_value': self.fill_value,
            'seed': self.seed,
        }
        base_config = super(ImageTransformation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


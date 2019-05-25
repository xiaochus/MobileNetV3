"""MobileNet v3 Large models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""


from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape
from keras.utils.vis_utils import plot_model

from model.mobilenet_base import MobileNetBase


class MobileNetV3_Large(MobileNetBase):
    def __init__(self, shape, n_class, alpha=1.0, include_top=True):
        """Init.

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if inculde classification layer.

        # Returns
            MobileNetv3 model.
        """
        super(MobileNetV3_Large, self).__init__(shape, n_class, alpha)
        self.include_top = include_top

    def build(self, plot=False):
        """build MobileNetV3 Large.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        inputs = Input(shape=self.shape)

        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

        x = self._bottleneck(x, 16, (3, 3), e=16, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=64, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=72, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=72, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck(x, 80, (3, 3), e=240, s=2, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=200, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 112, (3, 3), e=480, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 112, (3, 3), e=672, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=672, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')

        x = self._conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 960))(x)

        x = Conv2D(1280, (1, 1), padding='same')(x)
        x = self._return_activation(x, 'HS')

        if self.include_top:
            x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
            x = Reshape((self.n_class,))(x)

        model = Model(inputs, x)

        if plot:
            plot_model(model, to_file='images/MobileNetv3_large.png', show_shapes=True)

        return model

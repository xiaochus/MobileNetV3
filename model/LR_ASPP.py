"""Lite R-ASPP Semantic Segmentation based on MobileNetV3.
"""


from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Activation, Multiply, Add
from keras.utils.vis_utils import plot_model
from model.layers.bilinear_upsampling import BilinearUpSampling2D


class LiteRASSP:
    def __init__(self, input_shape, n_class=19, alpha=1.0, weights=None, backbone='small'):
        """Init.

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor (should be 1024 × 2048 or 512 × 1024 according 
                to the paper).
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier for mobilenetV3.
            weights: String, weights for mobilenetv3.
            backbone: String, name of backbone (must be small or large).
        """
        self.shape = input_shape
        self.n_class = n_class
        self.alpha = alpha
        self.weights = weights
        self.backbone = backbone

    def _extract_backbone(self):
        """extract feature map from backbone.
        """
        if self.backbone == 'large':
            from model.mobilenet_v3_large import MobileNetV3_Large

            model = MobileNetV3_Large(self.shape, self.n_class, alpha=self.alpha, include_top=False).build()
            layer_name8 = 'batch_normalization_13'
            layer_name16 = 'add_5'
        elif self.backbone == 'small':
            from model.mobilenet_v3_small import MobileNetV3_Small

            model = MobileNetV3_Small(self.shape, self.n_class, alpha=self.alpha, include_top=False).build()
            layer_name8 = 'batch_normalization_7'
            layer_name16 = 'add_2'
        else:
            raise Exception('Invalid backbone: {}'.format(self.backbone))

        if self.weights is not None:
            model.load_weights(self.weights, by_name=True)

        inputs= model.input
        # 1/8 feature map.
        out_feature8 = model.get_layer(layer_name8).output
        # 1/16 feature map.
        out_feature16 = model.get_layer(layer_name16).output

        return inputs, out_feature8, out_feature16

    def build(self, plot=False):
        """build Lite R-ASPP.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        inputs, out_feature8, out_feature16 = self._extract_backbone()

        # branch1
        x1 = Conv2D(128, (1, 1))(out_feature16)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        # branch2
        s = x1.shape

        x2 = AveragePooling2D(pool_size=(49, 49), strides=(16, 20))(out_feature16)
        x2 = Conv2D(128, (1, 1))(x2)
        x2 = Activation('sigmoid')(x2)
        x2 = BilinearUpSampling2D(target_size=(int(s[1]), int(s[2])))(x2)

        # branch3
        x3 = Conv2D(self.n_class, (1, 1))(out_feature8)

        # merge1
        x = Multiply()([x1, x2])
        x = BilinearUpSampling2D(size=(2, 2))(x)
        x = Conv2D(self.n_class, (1, 1))(x)

        # merge2
        x = Add()([x, x3])

        # out
        x = Activation('softmax')(x)

        model = Model(inputs=inputs, outputs=x)

        if plot:
            plot_model(model, to_file='images/LR_ASPP.png', show_shapes=True)

        return model

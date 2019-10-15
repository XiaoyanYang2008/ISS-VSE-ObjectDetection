import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, add, Input, AveragePooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2

class FasterRCNNModel:

    def __init__(self):

        print("instance initialized.")

    def resLyr(self, inputs,
               numFilters=16,
               kernelSize=3,
               strides=1,
               activation='relu',
               batchNorm=True,
               convFirst=True,
               lyrName=None):
        convLyr = Conv2D(numFilters,
                         kernel_size=kernelSize,
                         strides=strides,
                         padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4),
                         name=lyrName + '_conv' if lyrName else None)

        x = inputs
        if convFirst:
            x = convLyr(x)
            if batchNorm:
                x = BatchNormalization(name=lyrName + '_bn' if lyrName else None)(x)

            if activation is not None:
                x = Activation(activation, name=lyrName + '_' + activation if lyrName else None)(x)
        else:
            if batchNorm:
                x = BatchNormalization(name=lyrName + '_bn' if lyrName else None)(x)

            if activation is not None:
                x = Activation(activation, name=lyrName + '_' + activation if lyrName else None)(x)
            x = convLyr(x)

        return x

    def resBlkV1(self, inputs,
                 numFilters=16,
                 numBlocks=3,
                 kernelSize=3,
                 downSampleOnFirst=True,
                 names=None):
        x = inputs
        for run in range(0, numBlocks):
            strides = 1
            blkStr = str(run + 1)
            if downSampleOnFirst and run == 0:
                strides = 2

            y = self.resLyr(inputs=x,
                            numFilters=numFilters,
                            kernelSize=kernelSize,
                            strides=strides,
                            lyrName=names + '_Blk' + blkStr + '_Res1' if names else None)
            y = self.resLyr(inputs=y,
                            numFilters=numFilters,
                            kernelSize=kernelSize,
                            activation=None,
                            lyrName=names + '_Blk' + blkStr + '_Res2' if names else None)

            if downSampleOnFirst and run == 0:
                x = self.resLyr(inputs=x,
                                numFilters=numFilters,
                                kernelSize=1,
                                strides=strides,
                                activation=None,
                                batchNorm=False,
                                lyrName=names + '_Blk' + blkStr + '_lin' if names else None)

            x = add([x, y],
                    name=names + '_Blk' + blkStr + '_add' if names else None)

            x = Activation('relu', name=names + '_Blk' + blkStr + '_relu' if names else None)(x)

        return x

    def createResNetV1(self, inputShape=(128, 128, 3),
                       numberClasses=3):
        inputs = Input(shape=inputShape)
        v = self.resLyr(inputs, numFilters=16, kernelSize=5, lyrName='Inpt')
        v = self.resBlkV1(inputs=v,
                          numFilters=16,
                          numBlocks=5,
                          downSampleOnFirst=False,
                          names='Stg1')
        v = self.resBlkV1(inputs=v,
                          numFilters=32,
                          numBlocks=5,
                          downSampleOnFirst=True,
                          names='Stg2')
        v = self.resBlkV1(inputs=v,
                          numFilters=64,
                          numBlocks=5,
                          downSampleOnFirst=True,
                          names='Stg3')
        #     v = resBlkV1(inputs=v,
        #                  numFilters=512,
        #                  numBlocks=6,
        #                  downSampleOnFirst=True,
        #                  names='Stg4')
        v = AveragePooling2D(pool_size=8,
                             name='AvgPool')(v)
        v = Flatten()(v)
        outputs = Dense(numberClasses,
                        activation='softmax',
                        kernel_initializer='he_normal')(v)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(lr=0.002),
                      metrics=['accuracy'])
        return model

    def createFasterRCNNModel(self):
        resnet = self.createResNetV1()


    # start Training here.
    def train(self):
        print('Training starts.')

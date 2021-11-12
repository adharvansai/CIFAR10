### YOUR CODE HERE
# import tensorflow as tf
import torch
import torch.nn as nn
from torch.functional import Tensor
import torch.nn.functional as F

"""This script defines the network.
"""

class MyNetwork(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.network_size = self.configs['network_size']
        self.num_classes = self.configs['num_classes']
        self.first_num_filters = self.configs['first_num_filters']

        ### YOUR CODE HERE
        # define conv1
        self.start_layer = nn.Conv2d(3, 16, 3, padding=1, stride=1, bias=False)
        ### YOUR CODE HERE

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.

        block_fn = bottleneck_block

        self.stack_layers = nn.ModuleList()
        filters = self.first_num_filters
        for i in range(3):
            filters = self.first_num_filters * (2 ** i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(filters, block_fn, strides, self.network_size, self.first_num_filters))
        self.output_layer = output_layer(filters * 4, self.num_classes)

    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs


#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()
        ### YOUR CODE HERE
        self.nf = num_features
        self.eps_value = eps
        self.mom = momentum

        self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum)
        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        out = F.relu(self.batch_norm(inputs))
        return out

        ### YOUR CODE HERE



class bottleneck_block(nn.Module):
    """ Creates a bottleneck block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first
            convolution. NOTE: filters_out will be 4xfilters.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """

    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(bottleneck_block, self).__init__()

        self.first_layer = nn.Conv2d(filters, filters // 4, 1, padding=0)
        self.second_layer = nn.Conv2d(filters // 4, filters // 4, 3, padding=1)
        self.third_layer = nn.Conv2d(filters // 4, filters, 1, padding=0)
        self.residual_projection = None
        self.bnrelu_first = batch_norm_relu_layer(filters)
        self.bnrelu_rest = batch_norm_relu_layer(filters // 4)
        self.shortcut = nn.Sequential()

        if projection_shortcut is not None:
            self.residual_projection = nn.Conv2d(first_num_filters, filters, 1, stride=strides)
            self.first_layer = nn.Conv2d(first_num_filters, filters // 4, 1, stride=strides)
            self.bnrelu_first = batch_norm_relu_layer(first_num_filters)
            self.shortcut = nn.Sequential(nn.Conv2d(first_num_filters,filters,1,stride = strides, bias=False),
                                          batch_norm_relu_layer(filters))

        ### YOUR CODE HERE
        # Hint: Different from standard lib implementation, you need pay attention to
        # how to define in_channel of the first bn and conv of each block based on
        # Args given above.

        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        #print(inputs.size())
        preact_l1 = self.bnrelu_first(inputs)
        out_l1 = self.first_layer(preact_l1)
        preact_l2 = self.bnrelu_rest(out_l1)
        out_l2 = self.second_layer(preact_l2)
        preact_l3 = self.bnrelu_rest(out_l2)
        out_l3 = self.third_layer(preact_l3)

        res_out = out_l3 + self.shortcut(inputs)
        out = F.relu(res_out)

        return out

        ### YOUR CODE HERE
        #


class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
		block_fn: 'standard_block' or 'bottleneck_block'.
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
        network_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """

    def __init__(self, filters, block_fn, strides, network_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()
        filters_out = filters * 4
        ### END CODE HERE
        # projection_shortcut = ?
        # Only the first block per stack_layer uses projection_shortcut and strides
        blocks = []
        input_filters = first_num_filters
        for i in range(network_size):
            if (filters_out != input_filters):
                if(filters_out == 4*first_num_filters):
                    input_filters = first_num_filters
                    blocks.append(bottleneck_block(filters_out,'Y',strides, input_filters))
                    input_filters = filters_out
                else:
                    input_filters = filters_out//2
                    blocks.append(bottleneck_block(filters_out,'Y',strides, input_filters))
                    input_filters = filters_out
            else:
                blocks.append(bottleneck_block(filters_out, None, strides, filters_out))
        self.blocks = nn.Sequential(*blocks)

        ### END CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        out = self.blocks(inputs)
        return out

        ### END CODE HERE


class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """

    def __init__(self, filters, num_classes) -> None:
        super(output_layer, self).__init__()
        # Only apply the BN and ReLU for model that does pre_activation in each
        # bottleneck block, e.g. resnet V2.
        self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.997)

        ### END CODE HERE
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        input_filters = int(filters)

        self.fully_conn = nn.Linear(input_filters, num_classes)
        # self.sm = nn.Softmax(dim=1)
        ### END CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        pool_out = self.avg_pool(inputs)
        flat_out = pool_out.view(pool_out.size(0), -1)
        out_fc = self.fully_conn(flat_out)
        # out = self.sm(out_fc)
        return out_fc
        ### END CODE HERE
### END CODE HERE
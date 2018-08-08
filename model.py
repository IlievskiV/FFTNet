import numpy as np
import torch
from torch import nn


class FFTNetQueue(object):
    """
    
    """

    def __init__(self, batch_size, size, num_channels, cuda=True):
        super(FFTNetQueue, self).__init__()
        self.size = size
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.cuda = cuda
        self.queue = []
        self.reset()

    def reset(self):
        self.queue = torch.zeros([self.batch_size, self.num_channels, self.size])
        if self.cuda:
            self.queue = self.queue.cuda()

    def enqueue(self, x_push):
        x_pop = self.queue[:, :, -1].data
        self.queue[:, :, :-1] = self.queue[:, :, 1:]
        self.queue[:, :, -1] = x_push.view(x_push.shape[0], x_push.shape[1])
        return x_pop


class FFTNet(nn.Module):
    """
    A class representing the FFT Layer
    """

    def __init__(self, in_channels, out_channels, hid_channels, layer_id,
                 cond_channels=None, std_f=0.5):
        super(FFTNet, self).__init__()
        self.layer_id = layer_id
        self.receptive_field = 2 ** layer_id
        self.K = self.receptive_field // 2

        # the number of channels in the input
        self.in_channels = in_channels

        # the number of filters, or the number of channels in the last convolution output
        self.out_channels = out_channels

        # the number of filters, or the number of channels in the first convolution output
        self.hid_channels = hid_channels

        # the number of channels for the auxiliary input
        self.cond_channels = cond_channels

        # 1x1 convolution layer for the first half
        self.conv1_1 = nn.Conv1d(in_channels, hid_channels, 1, stride=1)

        # 1x1 convolution layer for the second half
        self.conv1_2 = nn.Conv1d(in_channels, hid_channels, 1, stride=1)

        # 1x1 convolution layer for the auxiliary input if any
        if cond_channels is not None:
            # 1x1 conv layer for the first half of the auxiliary vector
            self.convc1 = nn.Conv1d(cond_channels, hid_channels, 1)

            # 1x1 conv layer for the second half of the auxiliary vector
            self.convc2 = nn.Conv1d(cond_channels, hid_channels, 1)

        # 1x1 conv layer for the second convolution
        self.conv2 = nn.Conv1d(hid_channels, out_channels, 1)

        # ReLu layer
        self.relu = nn.ReLU()

        # initialize weights
        self.init_weights(std_f)

        # No idea
        self.buffer = None
        self.cond_buffer = None

        # inference params for linear operations
        self.w1_1 = None
        self.w1_2 = None
        self.w2 = None
        if cond_channels is not None:
            self.wc1_1 = None
            self.wc1_2 = None

    def init_weights(self, std_f):
        """
        
        :param std_f: 
        :return: 
        """
        std = np.sqrt(std_f / self.in_channels)
        self.conv1_1.weight.data.normal_(mean=0, std=std)
        self.conv1_1.bias.data.zero_()
        self.conv1_2.weight.data.normal_(mean=0, std=std)
        self.conv1_2.bias.data.zero_()
        if self.cond_channels is not None:
            self.convc1.weight.data.normal_(mean=0, std=std)
            self.convc1.bias.data.zero_()
            self.convc2.weight.data.normal_(mean=0, std=std)
            self.convc2.bias.data.zero_()

    def forward(self, x, cx=None):
        """
        
        
        Shapes:
            inputs: batch x channels x time
            cx: batch x cond_channels x time
            out: batch x out_chennels x time - receptive_field/2
        """
        T = x.shape[2]
        x1 = x[:, :, :-self.K]
        x2 = x[:, :, self.K:]
        z1 = self.conv1_1(x1)
        z2 = self.conv1_2(x2)
        z = z1 + z2
        # conditional input
        if cx is not None:
            cx1 = cx[:, :, :-self.K]
            cx2 = cx[:, :, self.K:]
            cz1 = self.convc1(cx1)
            cz2 = self.convc2(cx2)
            z = z + cz1 + cz2
        out = self.relu(z)
        out = self.conv2(out)
        out = self.relu(out)
        return out

    def forward_step(self, x, cx=None):
        """
        Forward pass only in inference time to speedup the inference.
        
        :param x: waveform
        :param cx: 
        :return: 
        """
        T = x.shape[2]
        B = x.shape[0]
        # linear weights
        if self.w1_1 is None:
            self.w1_1 = self._convert_to_fc_weights(self.conv1_1)
            self.w1_2 = self._convert_to_fc_weights(self.conv1_2)
        if cx is not None and self.wc1_1 is None:
            self.wc1_1 = self._convert_to_fc_weights(self.convc1)
            self.wc1_2 = self._convert_to_fc_weights(self.convc2)
        if self.w2 is None:
            self.w2 = self._convert_to_fc_weights(self.conv2)
        # create buffer queues
        if self.buffer is None:
            self.buffer = FFTNetQueue(B, self.K, self.in_channels, x.is_cuda)
        if self.cond_channels is not None and self.cond_buffer is None:
            self.cond_buffer = FFTNetQueue(B, self.K, self.cond_channels, x.is_cuda)
        # queue inputs
        x_input = x.view([B, -1])
        x_input2 = self.buffer.enqueue(x).view([B, -1])
        if self.cond_channels is not None:
            cx1 = cx.view([B, -1])
            cx2 = self.cond_buffer.enqueue(cx).view([B, -1])
        # perform first set of convs
        z1 = torch.nn.functional.linear(x_input, self.w1_1, self.conv1_1.bias)
        z2 = torch.nn.functional.linear(x_input2, self.w1_2, self.conv1_2.bias)
        z = z1 + z2
        if cx is not None:
            self.wc1_1 = self._convert_to_fc_weights(self.convc1)
            self.wc1_2 = self._convert_to_fc_weights(self.convc2)
            cz1 = torch.nn.functional.linear(cx1, self.wc1_1, self.convc1.bias)
            cz2 = torch.nn.functional.linear(cx2, self.wc1_2, self.convc2.bias)
            z = z + cz1 + cz2
        # second conv
        z = self.relu(z)
        z = torch.nn.functional.linear(z, self.w2, self.conv2.bias)
        z = self.relu(z)
        z = z.view(B, -1, 1)
        return z

    def _convert_to_fc_weights(self, conv):
        """
        It takes a convolutional layer and it transforms it to Fully-Connected.
        
        :param conv: 
        :return: 
        """
        w = conv.weight
        out_channels, in_channels, filter_size = w.shape
        nw = w.transpose(1, 2).view(out_channels, -1).contiguous()
        return nw


class FFTNetModel(nn.Module):
    """
    The entire FFT Network
    """

    def __init__(self, hid_channels=256, out_channels=256, n_layers=11,
                 cond_channels=None):
        super(FFTNetModel, self).__init__()
        self.cond_channels = cond_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.receptive_field = 2 ** n_layers

        self.layers = []
        for idx in range(self.n_layers):
            layer_id = n_layers - idx
            if idx == 0:
                layer = FFTNet(1, hid_channels, hid_channels, layer_id=layer_id, cond_channels=cond_channels)
            else:
                layer = FFTNet(hid_channels, hid_channels, hid_channels, layer_id=layer_id)
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)
        self.fc = nn.Linear(hid_channels, out_channels)

    def forward(self, x, cx=None):
        """
        Shapes:
            x: batch x 1 x time
            cx: batch x dim x time
        """

        # FFTNet modules
        out = x
        for idx, layer in enumerate(self.layers):
            if idx == 0 and cx is not None:
                out = layer(out, cx)
            else:
                out = layer(out)

        out = out.transpose(1, 2)
        out = self.fc(out)
        return out

    def forward_step(self, x, cx=None):
        """
        Only in inference to speedup the process.
        
        :param x: 
        :param cx: 
        :return: 
        """
        # FFTNet modules
        out = x
        for idx, layer in enumerate(self.layers):
            if idx == 0 and cx is not None:
                out = layer.forward_step(out, cx)
            else:
                out = layer.forward_step(out)
        out = out.transpose(1, 2)
        out = self.fc(out)
        return out


def sequence_mask(sequence_length):
    """
    Generates mask
    
    :param torch.Tensor sequence_length         : 1-D tensor for the length of each audio in the batch
    
    :return: mask showing which samples from one audio waveform are relevant, because they are all aligned
        to the same length, containing zeros at the end which should not be included in the loss
    
    
    I need to write an example of what is happening here.
    
    sequence_length = [2, 3, 3, 4, 5, 3, 6, 3, 2]
    
    max_len = 6
    
    batch_size = 9
    
    seq_range = [0, 1, 2, 3, 4, 5]
    seq_range_expand = 
        [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5]
        ]
        
    seq_length_expended = 
        [
            [2,  2,  2,  2,  2,  2]
            [3,  3,  3,  3,  3,  3]
            [3,  3,  3,  3,  3,  3]
            [4,  4,  4,  4,  4,  4]
            [5,  5,  5,  5,  5,  5]
            [3,  3,  3,  3,  3,  3]
            [6,  6,  6,  6,  6,  6]
            [3,  3,  3,  3,  3,  3]
            [2,  2,  2,  2,  2,  2]
        ]
    
    (seq_range_expand < seq_length_expand).float() = 
    
        [
            [ 1.,  1.,  0.,  0.,  0.,  0.],
            [ 1.,  1.,  1.,  0.,  0.,  0.],
            [ 1.,  1.,  1.,  0.,  0.,  0.],
            [ 1.,  1.,  1.,  1.,  0.,  0.],
            [ 1.,  1.,  1.,  1.,  1.,  0.],
            [ 1.,  1.,  1.,  0.,  0.,  0.],
            [ 1.,  1.,  1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  0.,  0.,  0.],
            [ 1.,  1.,  0.,  0.,  0.,  0.]
        ]
    
    
    """

    # determine the max len and the batch size
    max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)

    # this will generate a sequence of numbers starting from 0 to max_len
    seq_range = torch.arange(0, max_len).long()

    # first make the range a matrix of shape (1, max_len - 1) and then
    # expand it in a matrix of shape (batch_size, max_len)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()

    # expand it in the same shape as the previous matrix
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)

    return (seq_range_expand < seq_length_expand).float()


class MaskedCrossEntropyLoss(nn.Module):
    """
    
    """

    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

        # since reduce=False, it will return a loss pet batch element
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, input, target, lengths=None):
        """
        
        :param torch.Tensor input                   : the predictions from the last linear layer
        :param torch.Tensor target                  : the samples from the original audio waveform, shifted for one
        :param torch.Tensor lengths                 : 1-D tensor for the length of each audio in the batch
        
        :return: 
        """
        if lengths is None:
            raise RuntimeError(" > Provide lengths for the loss function")

        # the mask equals the maximal length
        mask = sequence_mask(lengths)
        if target.is_cuda:
            mask = mask.cuda()

        # transform the tensor to different shape such that, -1 means, the dimension will be inferred
        # the input from (batch_size, num_samples, quantization_levels) will be transformed to
        # (batch_size x num_samples, quantization_levels). This is a kind of flatenning.

        input = input.view([input.shape[0] * input.shape[1], -1])
        target = target.view([target.shape[0] * target.shape[1]])
        mask_ = mask.view([mask.shape[0] * mask.shape[1]])

        # calculate the cross entropy loss between the each row in the input and output
        # this is due to the reduce=False argument
        losses = self.criterion(input, target)

        # Returns the maximum value of each row of the input tensor in the given dimension dim = 1.
        # The second return value is the index location of each maximum value found (argmax).
        # in this case, the index of the quantization level in the prediction
        _, pred = torch.max(input, 1)

        # Convert tensor of boolean to float, where False = 0, True = 1
        # where the prediction and target do not match
        f = (pred != target).type(torch.FloatTensor)

        # where the prediction and target match
        t = (pred == target).type(torch.FloatTensor)

        # the XoR between f and t should give all 1's

        if input.is_cuda:
            f = f.cuda()
            t = t.cuda()

        f = (f.squeeze() * mask_).sum()
        t = (t.squeeze() * mask_).sum()

        # total loss, loss for false predicted, loss for correctly predicted
        return ((losses * mask_).sum()) / mask_.sum(), f.item(), t.item()


# https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4
# https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
class EMA(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta

    def assign_ema_model(self, model, new_model, cuda):
        new_model.load_state_dict(model.state_dict())
        for name, param in new_model.named_parameters():
            if name in self.shadow:
                param.data = self.shadow[name].clone()
        if cuda:
            new_model.cuda()
        return new_model

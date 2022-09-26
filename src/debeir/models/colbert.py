import logging
import os

import torch
from torch import nn
from transformers import BertModel, BertConfig
import json

logger = logging.getLogger(__name__)

ACT_FUNCS = {
    "relu": nn.ReLU,
}


LOSS_FUNCS = {
    'cross_entropy_loss': nn.CrossEntropyLoss,
}


class CoLBERTConfig(object):
    default_fname = "colbert_config.json"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__dict__.update(kwargs)

    def save(self, path, fname=default_fname):
        """
        :param fname: file name
        :param path: Path to save
        """
        json.dump(self.kwargs, open(os.path.join(path, fname), 'w+'))

    @classmethod
    def load(cls, path, fname=default_fname):
        """
        Load the ColBERT config from path (don't point to file name just directory)
        :return ColBERTConfig:
        """

        kwargs = json.load(open(os.path.join(path, fname)))

        return CoLBERTConfig(**kwargs)


class ConvolutionalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, first_stride=1, act_func=nn.ReLU):
        super(ConvolutionalBlock, self).__init__()

        padding = int((kernel_size - 1) / 2)
        if kernel_size == 3:
            assert padding == 1  # checks
        if kernel_size == 5:
            assert padding == 2  # checks
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=first_stride, padding=padding),
            nn.BatchNorm1d(num_features=out_channels)
        ]

        if act_func is not None:
            layers.append(act_func())

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class KMaxPool(nn.Module):
    def __init__(self, k=1):
        super(KMaxPool, self).__init__()

        self.k = k

    def forward(self, x):
        # x : batch_size, channel, time_steps
        if self.k == 'half':
            time_steps = x.shape(2)
            self.k = time_steps // 2

        kmax, kargmax = torch.topk(x, self.k, sorted=True)
        # kmax, kargmax = x.topk(self.k, dim=2)
        return kmax


def visualisation_dump(argmax, input_tensors):
    pass

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, optional_shortcut=True,
                 kernel_size=1, act_func=nn.ReLU):
        super(ResidualBlock, self).__init__()
        self.optional_shortcut = optional_shortcut
        self.convolutional_block = ConvolutionalBlock(in_channels, out_channels, first_stride=1,
                                                      act_func=act_func, kernel_size=kernel_size)

    def forward(self, x):
        residual = x
        x = self.convolutional_block(x)

        if self.optional_shortcut:
            x = x + residual

        return x


class ColBERT(nn.Module):
    def __init__(self, bert_model_args, bert_model_kwargs, config: BertConfig, device: str, max_seq_len: int
            = 128, k: int = 8,
                 optional_shortcut: bool = True, hidden_neurons: int = 2048, use_batch_norms: bool = True,
                 use_trans_blocks: bool = False, residual_kernel_size: int = 1, dropout_perc: float = 0.5,
                 act_func="mish", loss_func='cross_entropy_loss', **kwargs):  # kwargs for compat

        super().__init__()
        self.device = device
        hidden_dim = config.hidden_size
        self.seq_length = max_seq_len
        self.use_trans_blocks = use_trans_blocks
        self.use_batch_norms = use_batch_norms
        self.num_layers = config.num_hidden_layers
        num_labels = config.num_labels
        self.loss_func = LOSS_FUNCS[loss_func.lower()]()

        # Save our kwargs to reinitialise the model during evaluation
        self.bert_config = config
        self.colbert_config = CoLBERTConfig(k=k,
                                            optional_shortcut=optional_shortcut, hidden_neurons=hidden_neurons,
                                            use_batch_norms=use_batch_norms, use_trans_blocks=use_trans_blocks,
                                            residual_kernel_size=residual_kernel_size, dropout_perc=dropout_perc,
                                            act_func=act_func, bert_model_args=bert_model_args,
                                            bert_model_kwargs=bert_model_kwargs)

        logging.info("ColBERT Configuration %s" % str(self.colbert_config.kwargs))

        # relax this constraint later
        assert act_func.lower() in ACT_FUNCS, f"Error not in activation function dictionary, {ACT_FUNCS.keys()}"
        act_func = ACT_FUNCS[act_func.lower()]

        # CNN Part
        conv_layers = []
        transformation_blocks = [None]  # Pad the first element, for the for loop in forward
        batch_norms = [None]  # Pad the first element

        # Adds up to num_layers + 1 embedding layer
        conv_layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1))

        for i in range(self.num_layers):
            # Create the residual blocks, batch_norms and transformation blocks

            conv_layers.append(ResidualBlock(hidden_dim, hidden_dim, optional_shortcut=optional_shortcut,
                                             kernel_size=residual_kernel_size, act_func=act_func))
            if use_trans_blocks:
                transformation_blocks.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1))
            if use_batch_norms:
                batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.conv_layers = nn.ModuleList(conv_layers)

        if use_trans_blocks:
            self.transformation_blocks = nn.ModuleList(transformation_blocks)
        if use_batch_norms:
            self.batch_norms = nn.ModuleList(batch_norms)

        self.kmax_pooling = KMaxPool(k)

        # Create the MLP to compress the k signals
        linear_layers = list()
        linear_layers.append(nn.Linear(hidden_dim * k, num_labels))  # Downsample into Kmaxpool?
        #linear_layers.append(nn.Linear(hidden_neurons, hidden_neurons))
        #linear_layers.append(nn.Dropout(dropout_perc))
        #linear_layers.append(nn.Linear(hidden_neurons, num_labels))

        self.linear_layers = nn.Sequential(*linear_layers)
        self.apply(weight_init)
        self.bert = BertModel.from_pretrained(*bert_model_args, **bert_model_kwargs,
                config=self.bert_config)  # Add Bert model after random initialisation

        for param in self.bert.pooler.parameters(): # We don't need the pooler
            param.requires_grad = False

        self.bert.to(self.device)

    def forward(self, *args, **kwargs):
        # input_ids: batch_size x seq_length x hidden_dim
        labels = kwargs['labels'] if 'labels' in kwargs else None
        if labels is not None: del kwargs['labels']

        bert_outputs = self.bert(*args, **kwargs)
        hidden_states = bert_outputs[-1]

        # Fix this, also draw out what ur model should do first
        is_embedding_layer = True

        assert len(self.conv_layers) == len(
            hidden_states)  # == len(self.transformation_blocks) == len(self.batch_norms), info

        zip_args = [self.conv_layers, hidden_states]
        identity = lambda k: k

        if self.use_trans_blocks:
            assert len(self.transformation_blocks) == len(hidden_states)
            zip_args.append(self.transformation_blocks)
        else:
            zip_args.append([identity for i in range(self.num_layers+1)])

        if self.use_batch_norms:
            assert len(self.batch_norms) == len(hidden_states)
            zip_args.append(self.batch_norms)
        else:
            zip_args.append([identity for i in range(self.num_layers+1)])

        out = None
        for co, hi, tr, bn in zip(*zip_args):
            if is_embedding_layer:
                out = co(hi.transpose(1, 2))  # batch x hidden x seq_len
                is_embedding_layer = not is_embedding_layer
            else:
                out = co(out + tr(bn(hi.transpose(1, 2))))  # add hidden dims together

        assert out.shape[2] == self.seq_length

        out = self.kmax_pooling(out)
        # batch_size x seq_len x hidden -> batch_size x flatten
        logits = self.linear_layers(torch.flatten(out, start_dim=1))

        return self.loss_func(logits, labels), logits


    @classmethod
    def from_config(cls, *args, config_path):
        kwargs = torch.load(config_path)
        return ColBERT(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, output_dir, **kwargs):
        config_found = True
        colbert_config = None

        try:
            colbert_config = CoLBERTConfig.load(output_dir)
        except:
            config_found = False

        bert_config = None

        if 'config' in kwargs:
            bert_config = kwargs['config']
            del kwargs['config']
        else:
            bert_config = BertConfig.from_pretrained(output_dir)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = None

        if config_found:
            model = ColBERT(config=bert_config, device=device, **colbert_config.kwargs)
            model.load_state_dict(torch.load(output_dir + '/cnn_bert.pth'))
            logger.info(f"*** Loaded CNN Bert Model Weights from {output_dir + '/cnn_bert.pth'}")

        else:
            model = ColBERT((output_dir,), {}, config=bert_config, **kwargs)
            logger.info(f"*** Create New CNN Bert Model ***")

        return model

    def save_pretrained(self, output_dir):
        logger.info(f"*** Saved Bert Model Weights to {output_dir}")
        self.bert.save_pretrained(output_dir)
        torch.save(self.state_dict(), output_dir + '/cnn_bert.pth')
        self.bert_config.save_pretrained(output_dir)
        self.colbert_config.save(output_dir)
        logger.info(f"*** Saved CNN Bert Model Weights to {output_dir + '/cnn_bert.pth'}")


class ComBERT(nn.Module):
    def __init__(self, bert_model_args, bert_model_kwargs, config: BertConfig, device: str, max_seq_len: int= 128,
                 k: int = 8, optional_shortcut: bool = True, hidden_neurons: int = 2048, use_batch_norms: bool = True,
                 use_trans_blocks: bool = False, residual_kernel_size: int = 1, dropout_perc: float = 0.5,
                 act_func="mish", loss_func='cross_entropy_loss', num_blocks=2, **kwargs):  # kwargs for compat

        super().__init__()
        self.device = device
        hidden_dim = config.hidden_size
        self.seq_length = max_seq_len
        self.use_trans_blocks = use_trans_blocks
        self.use_batch_norms = use_batch_norms
        self.num_layers = config.num_hidden_layers
        num_labels = config.num_labels
        self.num_blocks = num_blocks
        self.loss_func = LOSS_FUNCS[loss_func.lower()]()

        # Save our kwargs to reinitialise the model during evaluation
        self.bert_config = config
        self.colbert_config = CoLBERTConfig(k=k,
                                            optional_shortcut=optional_shortcut, hidden_neurons=hidden_neurons,
                                            use_batch_norms=use_batch_norms, use_trans_blocks=use_trans_blocks,
                                            residual_kernel_size=residual_kernel_size, dropout_perc=dropout_perc,
                                            act_func=act_func, bert_model_args=bert_model_args,
                                            bert_model_kwargs=bert_model_kwargs)

        logging.info("ColBERT Configuration %s" % str(self.colbert_config.kwargs))

        # relax this constraint later
        assert act_func.lower() in ACT_FUNCS, f"Error not in activation function dictionary, {ACT_FUNCS.keys()}"
        act_func = ACT_FUNCS[act_func.lower()]

        # CNN Part
        conv_layers = []

        # Adds up to num_layers + 1 embedding layer
        conv_layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1))


        for i in range(num_blocks):
            conv_layers.append(ResidualBlock(hidden_dim, hidden_dim, optional_shortcut=optional_shortcut,
                                             kernel_size=residual_kernel_size, act_func=act_func))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.kmax_pooling = KMaxPool(k)

        # Create the MLP to compress the k signals
        linear_layers = list()
        linear_layers.append(nn.Linear(hidden_dim * k, hidden_neurons))  # Downsample into Kmaxpool?
        linear_layers.append(nn.Dropout(dropout_perc))
        linear_layers.append(nn.Linear(hidden_neurons, num_labels))

        self.linear_layers = nn.Sequential(*linear_layers)
        self.apply(weight_init)
        self.bert = BertModel.from_pretrained(*bert_model_args, **bert_model_kwargs,
                                              config=self.bert_config)  # Add Bert model after random initialisation
        self.bert.to(self.device)

    def forward(self, *args, **kwargs):
        # input_ids: batch_size x seq_length x hidden_dim

        labels = kwargs['labels'] if 'labels' in kwargs else None
        if labels is not None: del kwargs['labels']

        bert_outputs = self.bert(*args, **kwargs)
        hidden_states = list(bert_outputs[-1])
        embedding_layer = hidden_states.pop(0)

        split_size = len(hidden_states) // self.num_blocks

        assert split_size % 2 == 0, "must be an even number"
        split_layers = [hidden_states[x:x+split_size] for x in range(0, len(hidden_states), split_size)]
        split_layers.insert(0, embedding_layer)

        assert len(self.conv_layers) == len(split_layers), "must have equal inputs in length"

        outputs = []

        for cnv, layer in zip(self.conv_layers, split_layers):
            outputs.append(self.kmax_pooling(cnv(layer)))

        # batch_size x seq_len x hidden -> batch_size x flatten
        logits = self.linear_layers(torch.flatten(torch.cat(outputs, dim=-1), start_dim=1))

        return self.loss_func(logits, labels), logits


    @classmethod
    def from_config(cls, *args, config_path):
        kwargs = torch.load(config_path)
        return ComBERT(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, output_dir, **kwargs):
        config_found = True
        colbert_config = None

        try:
            colbert_config = CoLBERTConfig.load(output_dir)
        except:
            config_found = False

        bert_config = None

        if 'config' in kwargs:
            bert_config = kwargs['config']
            del kwargs['config']
        else:
            bert_config = BertConfig.from_pretrained(output_dir)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = None

        if config_found:
            model = ComBERT(config=bert_config, device=device, **colbert_config.kwargs)
            model.load_state_dict(torch.load(output_dir + '/cnn_bert.pth'))
            logger.info(f"*** Loaded CNN Bert Model Weights from {output_dir + '/cnn_bert.pth'}")

        else:
            model = ComBERT((output_dir,), {}, config=bert_config, **kwargs)
            logger.info(f"*** Create New CNN Bert Model ***")

        return model

    def save_pretrained(self, output_dir):
        logger.info(f"*** Saved Bert Model Weights to {output_dir}")
        self.bert.save_pretrained(output_dir)
        torch.save(self.state_dict(), output_dir + '/cnn_bert.pth')
        self.bert_config.save_pretrained(output_dir)
        self.colbert_config.save(output_dir)
        logger.info(f"*** Saved CNN Bert Model Weights to {output_dir + '/cnn_bert.pth'}")
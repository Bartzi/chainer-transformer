import chainer
import chainer.functions as F
import chainer.links as L

from chainer import Chain

from transformer import get_encoder_decoder
from transformer.utils import subsequent_mask


class CopyTransformer(Chain):

    def __init__(self, vocab_size, max_len, start_symbol, transformer_size=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.start_symbol = start_symbol
        self.transformer_size = transformer_size

        with self.init_scope():
            model = get_encoder_decoder(
                vocab_size,
                vocab_size,
                N=2,
                model_size=transformer_size,
            )
            self.model = model
            self.mask = subsequent_mask(self.transformer_size)
            self.classifier = L.Linear(transformer_size, vocab_size)

    def __call__(self, x, t):
        result = self.model(x, t, None, self.mask)

        return self.classifier(result, n_batch_axes=2)

    def decode_prediction(self, x):
        return F.argmax(F.softmax(x, axis=2), axis=2)

    def predict(self, x):
        memory = self.model.encode(x, None)

        target = self.xp.full((len(x), 1), self.start_symbol, x.dtype)

        for i in range(self.max_len - 1):
            prediction = self.model.decode(memory, None, target, self.mask)
            prediction = self.classifier(prediction, n_batch_axes=2)
            decoded = self.decode_prediction(prediction)

            target = F.concat([target, decoded[:, -1:]])

        return target.array

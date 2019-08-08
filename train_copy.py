import argparse

import chainer

from chains.copy_transformer import CopyTransformer
from datasets.copy_dataset import CopyDataset
from evaluation.copy_transformer_eval_function import CopyTransformerEvaluationFunction, CopyTransformerEvaluator
from hooks.noam_hook import NoamOptimizer
from updater.copy_transformer_updater import CopyTransformerUpdater


def main(args):
    vocab_size = 101
    train_dataset = CopyDataset(vocab_size)
    val_dataset = CopyDataset(vocab_size)

    train_iter = chainer.iterators.MultithreadIterator(train_dataset, args.batch_size)
    val_iter = chainer.iterators.MultithreadIterator(val_dataset, args.batch_size, shuffle=False, repeat=False)

    net = CopyTransformer(vocab_size, train_dataset.max_len, train_dataset.start_symbol)

    optimizer = chainer.optimizers.Adam(alpha=0, beta1=0.9, beta2=0.98, eps=1e-9)
    optimizer.setup(net)
    optimizer.add_hook(
        NoamOptimizer(4000, 2, net.transformer_size)
    )

    updater = CopyTransformerUpdater(train_iter, optimizer, device=args.gpu)

    trainer = chainer.training.Trainer(updater, (args.epochs, 'epoch'), out='train')
    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.PrintReport(['epoch', 'loss', 'train/accuracy', 'part_accuracy', 'accuracy']))
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))

    eval_function = CopyTransformerEvaluationFunction(net, args.gpu)

    trainer.extend(CopyTransformerEvaluator(val_iter, net, device=args.gpu, eval_func=eval_function))

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the transformer under a copy task")
    parser.add_argument("-g", "--gpu", default=-1, type=int, help="gpu device to use (negative value indicates cpu)")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size for training")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs to train")

    args = parser.parse_args()
    main(args)


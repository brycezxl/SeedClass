import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=True,
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate the model on dev set')
    parser.add_argument('--model-saved-path', type=str, default='results/model',
                        help='')
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='train the model')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='')
    parser.add_argument('--max-num-epochs', type=int, default=50,
                        help='')
    parser.add_argument('--d-model', type=int, default=512,
                        help='')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    return parser.parse_args()


def main(args):
    print(args)
    from runner import Runner
    runner = Runner(args)
    if args.train:
        runner.train()
    if args.evaluate:
        runner.eval()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main(parse_args())

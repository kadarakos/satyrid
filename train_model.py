"""
Example execution script. The dataset parameter can
be modified to coco/flickr30k/flickr8k
"""
import argparse

from capgen import train

parser = argparse.ArgumentParser()
parser.add_argument("--attn_type",  default="deterministic",
                    help="type of attention mechanism")
parser.add_argument("changes",  nargs="*",
                    help="Changes to default values", default="")


def main(params):
    # see documentation in capgen.py for more details on hyperparams
    _, validerr, _ = train(saveto=params["model"],
                           attn_type=params["attn-type"],
                           reload_=params["reload"],
                           dim_word=params["dim-word"],
                           ctx_dim=params["ctx-dim"],
                           dim=params["dim"],
                           n_layers_att=params["n-layers-att"],
                           n_layers_out=params["n-layers-out"],
                           n_layers_lstm=params["n-layers-lstm"],
                           n_layers_init=params["n-layers-init"],
                           n_words=params["n-words"],
                           lstm_encoder=params["lstm-encoder"],
                           decay_c=params["decay-c"],
                           alpha_c=params["alpha-c"],
                           prev2out=params["prev2out"],
                           ctx2out=params["ctx2out"],
                           lrate=params["learning-rate"],
                           optimizer=params["optimizer"],
                           selector=params["selector"],
                           patience=10,
                           maxlen=100,
                           batch_size=params['batch_size'],
                           valid_batch_size=64,
                           validFreq=params['validFreq'],
                           dispFreq=100,
                           saveFreq=1000,
                           sampleFreq=params['sampleFreq'],
                           dataset=params['dataset'],
                           use_dropout=params["use-dropout"],
                           use_dropout_lstm=params["use-dropout-lstm"],
                           save_per_epoch=params["save-per-epoch"],
                           clipnorm=params['clipnorm'],
                           clipvalue=params['clipvalue'],
                           references=params['references'])
    print("Best valid cost: %.2f}" % validerr)


if __name__ == "__main__":
    # stochastic == hard attention
    # deterministic == soft attention
    defaults = {"model": "coco-soft_adam_dropout-dim1800-1e-4.npz",
                #"attn-type": "stochastic",
                "attn-type": "deterministic",
                "dim-word": 512,
                "ctx-dim": 512,
                "dim": 1800,
                "n-words": 9584,
                "n-layers-att": 2,
                "n-layers-out": 1,
                "n-layers-lstm": 1,
                "n-layers-init": 2,
                "lstm-encoder": False,
                "decay-c": 1e-8, # L2 regularisation
                "alpha-c": 1., # doubly-stochastic regularisation
                "prev2out": True,
                "ctx2out": True,
                "learning-rate": 0.0001, # not used for adadelta
                "optimizer": "adam",
                "selector": True,
                "use-dropout": True,
                "use-dropout-lstm": False,
                "save-per-epoch": False,
                "reload": False,
                "dispFreq": 5000, # show 5 samples from the model
                "validFreq": 500, # check loss on validation data
                "sampleFreq": 5000,
                "batch_size": 64,
                "clipnorm": 4.,
                "clipvalue": 0.,
                "references": "ref/coco/dev/",
                "dataset": "coco" # dataset module
                }
    # get updates from command line
    args = parser.parse_args()
    for change in args.changes:
        defaults.update(eval("dict({})".format(change)))
    main(defaults)

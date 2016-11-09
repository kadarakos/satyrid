"""
Example execution script. The dataset parameter can
be modified to coco/flickr30k/flickr8k
"""
import argparse

from model import train

parser = argparse.ArgumentParser()
parser.add_argument("--attn_type",  default="deterministic",
                    help="type of attention mechanism")
parser.add_argument("changes",  nargs="*",
                    help="Changes to default values", default="")


def main(params):
    # see documentation in model.py for more details on hyperparams
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
                           max_epochs=params['max_epochs'],
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
                           references=params['references'],
                           use_metrics=params['use_metrics'],
                           metric=params['metric'],
                           force_metrics=params['force_metrics'])
    print("Average valid cost: {:.2f}".format(validerr.mean()))


if __name__ == "__main__":
    # "attn-type": "stochastic" == hard attention
    # "attn-type": "deterministic: == soft attention
    defaults = {"model": "flickr30k-soft_attn-w512-h1300-nosal.npz",
                "dataset": "flickr30k", # dataset module
                "references": "ref/30k/dev/",
                #"attn-type": "stochastic", # hard attention
                "attn-type": "deterministic", # soft attention
                "dim-word": 512,
                "ctx-dim": 512,
                "dim": 1300,
                "n-words": 10000,
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
                "max_epochs": 50,
                "selector": True,
                "use-dropout": True,
                "use-dropout-lstm": False,
                "save-per-epoch": False,
                "reload": False,
                "dispFreq": 1, # updates between showing 5 samples from the model
                "validFreq": 1000, # updates between validation data loss check
                "sampleFreq": 1000,
                "batch_size": 64,
                "clipnorm": 4., # value to clip the norm of the gradients
                "clipvalue": 0., # value to clip the value of the updates
                "use_metrics": True, # measure training progress using metrics
                "metric": "Bleu_4", # METEOR, Bleu_{1,2,3,4}, ROUGE_L, CIDEr
                # By default, metrics are computed at the end of each epoch.
                # Do you want to also compute metrics when val loss decreases?
                # This will substantially increase training time!
                "force_metrics": True,
		"saliency": False,
		"lamb": 0.06
                }
    # get updates from command line
    args = parser.parse_args()
    for change in args.changes:
        defaults.update(eval("dict({})".format(change)))
    main(defaults)

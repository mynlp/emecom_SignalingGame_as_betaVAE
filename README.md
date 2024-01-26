# Lewis's Signaling Game as beta-VAE For Natural Word Lengths and Segments

This is the implementation code of:

R. Ueda & T. Taniguchi.
[Lewis's Signaling Game as beta-VAE For Natural Word Lengths and Segments](https://openreview.net/forum?id=HC0msxE3sf).
ICLR 2024.

## Reproducing the Experiments
```
./venv/bin/python -m emecom_gen.zoo.attval_signaling_game.train \
    --vocab_size 9 \
    --max_len 32 \
    --fix_message_length false \
    --n_epochs 20000 \
    --batch_size 8192 \
    --sender_cell_type gru \
    --sender_hidden_size 512 \
    --sender_embedding_dim 32 \
    --sender_layer_norm true \
    --receiver_cell_type gru \
    --receiver_hidden_size 512 \
    --receiver_embedding_dim 32 \
    --receiver_layer_norm true \
    --receiver_dropout_alpha 1e-3 \
    --baseline_type baseline-from-sender \
    --beta_scheduler_type rewo
    --beta_rewo_communication_loss constraint 0.3 \
    --prior_type receiver \
    --n_attributes ${N_ATTRIBUTES} \
    --n_values ${N_VALUES} \
    --random_seed ${RANDOM_SEED}

```
## Citation
```
@inproceedings{ueda2024lewis,
    title={Lewis's Signaling Game as beta-{VAE} For Natural Word Lengths and Segments},
    author={Ryo Ueda and Tadahiro Taniguchi},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=HC0msxE3sf}
}
```
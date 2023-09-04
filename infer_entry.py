import torch
from torch import nn
from treevalue import FastTreeValue

from gende import tokenize
from gende.models import load_model_from_ckpt


class ModelWithEncoding(nn.Module):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model

    def forward(self, x):
        encoded = self.model.encoder(x)
        predicted = self.model.ft(encoded)

        return FastTreeValue({
            'encoded': encoded,
            'predicted': predicted,
        })


_torch_stack = FastTreeValue.func(subside=True)(torch.stack)

if __name__ == '__main__':
    # use your ckpt here!
    ckpt_file = 'runs/bert_mean_seed_0/ckpts/best.ckpt'
    m = load_model_from_ckpt(ckpt_file)
    m = ModelWithEncoding(m)
    print(m)

    with torch.no_grad():
        # batch input
        input_ = _torch_stack([
            tokenize(
                "The Peashooter is your initial line of defense in Plants vs. Zombies. With a cooldown time of 7.5 "
                "seconds, it can deal 20 points of damage to attacking zombies. Its attack interval is 1.4 seconds, "
                "and it has a health of 300. The Peashooter costs 100 units of sunlight to plant. Its primary "
                "function is shooting peas at zombies to eliminate them.", tokenizer_name='bert',
            ),
            tokenize(
                "Sunflowers are crucial for generating additional sunlight, which is necessary to plant other plants. "
                "They have a cooldown time of 7.5 seconds and provide 25 units of sunlight every 24 seconds. With a "
                "health of 300 and a cost of 50 units of sunlight, Sunflowers play a supportive role by providing the "
                "resources needed to expand your defense.", tokenizer_name='bert',
            ),
        ])

        print(input_)
        output = m(input_)
        print(output)

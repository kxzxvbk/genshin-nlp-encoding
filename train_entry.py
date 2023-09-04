from ditk import logging

from gende.train import train

if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    train(
        'runs/bert_mean_seed_0',
        model_name='bert_mean',
        datasource='./data/annotation.xlsx',
        seed=0,
        max_epochs=50,
        train_eval_all=True
    )

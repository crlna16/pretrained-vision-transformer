#!/usr/bin/env python

import sys
import yaml
import pprint

from torch.utils.data import DataLoader
import lightning as L

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

sys.path.append('./src/')
from country211_module import Country211DataModule
from vision_transformer import VisionTransformerPretrained

import utils


def main(arg):
    L.seed_everything(1312)
    print('Seeed everything')

    with open('./configs/default.yaml') as cf_file:
        default_config = yaml.safe_load(cf_file.read())

    print(len(arg))

    if len(arg) == 1:
        with open(arg[0]) as cf_file:
            config = yaml.safe_load( cf_file.read() )

        config = utils.merge_dictionaries_recursively(default_config, config)
        print(f'Configuration read from {arg[0]}')
    else:
        config = default_config
        print('Configuration read from ./configs/default.yaml')

    print()
    print('Configuration:')
    pprint.pprint(config)
    print()

    # setup data
    datamodule = Country211DataModule(config['data_root'], config['batch_size'])
    datamodule.prepare_data()
    datamodule.setup('train')

    train_dataloader = datamodule.train_dataloader()
    valid_dataloader = datamodule.valid_dataloader()
    test_dataloader = datamodule.test_dataloader()

    # setup model
    model = VisionTransformerPretrained(config['model'], datamodule.num_classes)

    # setup callbacks
    early_stopping = EarlyStopping(monitor='valid_acc', patience=5, mode='max')

    # logger
    logger = TensorBoardLogger("tensorboard_logs", name=config['run_id'])

    # train
    trainer = L.Trainer(callbacks=[early_stopping], logger=logger, enable_progress_bar=False)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # test
    trainer.test(model=model, dataloaders=test_dataloader, verbose=True)






if __name__=='__main__':
    main(sys.argv[1:])

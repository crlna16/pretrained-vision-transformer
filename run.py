#!/usr/bin/env python

import sys
import os
import yaml
import pprint

from torch.utils.data import DataLoader
import lightning as L

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

sys.path.append('./src/')
from country211_module import Country211DataModule
from eurosat_module import EuroSAT_RGB_DataModule
from vision_transformer import VisionTransformerPretrained

import utils


def main(arg):
    L.seed_everything(1312)
    print('Seeed everything')

    with open('./configs/default.yaml') as cf_file:
        default_config = yaml.safe_load(cf_file.read())

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
    path_to_data = os.path.join(config['data_root'], config['data_set'])

    if config['data_set'] == 'Country211':
        datamodule = Country211DataModule(path_to_data, config['batch_size'])
    elif config['data_set'] == 'EuroSAT_RGB':
        datamodule = EuroSAT_RGB_DataModule(path_to_data, config['batch_size'], config['valid_size'])
        
    datamodule.prepare_data()
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    valid_dataloader = datamodule.valid_dataloader()
    test_dataloader = datamodule.test_dataloader()

    print('Finished data loading')
    print('Number of classes:', datamodule.num_classes)

    # setup model
    model = VisionTransformerPretrained(config['model'], datamodule.num_classes, config['learning_rate'])
    print('Finished model loading')

    # setup callbacks
    early_stopping = EarlyStopping(monitor='valid_acc', patience=config['early_stopping_patience'], mode='max')

    # logger
    logger = TensorBoardLogger("tensorboard_logs", name=config['run_id'])

    # train
    trainer = L.Trainer(max_epochs=config['max_epochs'],
                        devices=config['devices'], 
                        callbacks=[early_stopping], 
                        logger=logger,
                        enable_progress_bar=False)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # test
    print('Starting model test')
    trainer.test(model=model, dataloaders=test_dataloader, verbose=True)
    print('Finished model test')






if __name__=='__main__':
    main(sys.argv[1:])

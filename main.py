import argparse
import os

from torch.utils.data import DataLoader
import torch

from config import cfg
from data import ToneNetDatasetWordEmbeddingRhythm
from model import ToneNetWithWordEmbeddingWithRhythm
from tools import do_train, do_test
from utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="ToneNet Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("ToneNet Training", output_dir)
    logger.info(args)

    if args.config_file != "":
        logger.info(f"Loaded configuration file {args.config_file}")
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info(f"Running with config:\n{cfg}")

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_data = torch.load(cfg.DATASETS.DATA)
    test_data = torch.load(cfg.TEST.DATA)
    all_S_data = torch.load(cfg.DATASETS.DATA)

    train_dataset = ToneNetDatasetWordEmbeddingRhythm(
        data=train_data,
        all_S_data=all_S_data,
        used_rhythm_feature_nums=9,
        random_jointed_s=False,
    )
    test_dataset = ToneNetDatasetWordEmbeddingRhythm(
        data=test_data,
        all_S_data=all_S_data,
        used_rhythm_feature_nums=9,
        random_jointed_s=False,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_size=cfg.SOLVER.SEQUENCE_PER_BATCH,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_size=cfg.SOLVER.SEQUENCE_PER_BATCH,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
    )
    model = ToneNetWithWordEmbeddingWithRhythm(d_rhythm_in=9, d_model=256, nhead=4, dropout_rate=0.1)
    if cfg.MODEL.PRETRAIN_PATH != '':
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH, map_location='cpu')['model'])

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    scheduler=None

    if 'cpu' not in cfg.MODEL.DEVICE:
        model.to(device=cfg.MODEL.DEVICE)

    if cfg.TEST.EVALUATE_ONLY == 'on':
        loss_test, _, performance_test_str = do_test(
            cfg=cfg,
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            epoch=cfg.SOLVER.MAX_EPOCHS,
        )
        logger.info(f"Test loss:{loss_test:.3f}")
        logger.info(performance_test_str)
    else:
        do_train(
            cfg=cfg,
            model=model,
            dataloader_train=train_dataloader,
            dataloader_test=test_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            logger=logger
        )


if __name__ == '__main__':
    main()

import os
import sys
import logging
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from tqdm import tqdm

sys.path.insert(0, '/home/o_a38510/sn-spotting/Benchmarks/CALF/src')

from dataset import SoccerNetClipsTesting
from model import ContextAwareModel
from train import test

torch.manual_seed(0)
np.random.seed(0)


def main(args):

    sn_dir = '/home/o_a38510/sn-spotting'
    os.chdir(sn_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("=" * 80)
    logging.info("Testing CALF on SoccerNet test set")
    logging.info("=" * 80)
    logging.info(f"Dataset path: {args.SoccerNet_path}")
    logging.info(f"Features: {args.features}")
    logging.info(f"Model: {args.model_name}")
    logging.info("=" * 80)

    logging.info("loading test dataset...")
    dataset_Test = SoccerNetClipsTesting(
        path=args.SoccerNet_path,
        features=args.features,
        split="test",
        framerate=args.framerate,
        chunk_size=args.chunk_size * args.framerate,
        receptive_field=args.receptive_field * args.framerate
    )

    logging.info(f"loaded {len(dataset_Test)} games")

    test_loader = torch.utils.data.DataLoader(
        dataset_Test,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    logging.info("setting up model...")
    model = ContextAwareModel(
        weights=None,
        input_size=args.num_features,
        num_classes=dataset_Test.num_classes,
        chunk_size=args.chunk_size * args.framerate,
        dim_capsule=args.dim_capsule,
        receptive_field=args.receptive_field * args.framerate,
        num_detections=dataset_Test.num_detections,
        framerate=args.framerate
    ).cuda()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"model has {params} trainable params")

    model_path = os.path.join(
        "/home/o_a38510/sn-spotting/Benchmarks/CALF/models",
        args.model_name,
        "model.pth.tar"
    )

    if not os.path.exists(model_path):
        logging.error(f"model not found at {model_path}")
        sys.exit(1)

    logging.info(f"loading weights from {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    logging.info("starting inference...")
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, \
    a_mAP_unshown, a_mAP_per_class_unshown = test(
        test_loader,
        model=model,
        model_name=args.model_name,
        save_predictions=True
    )

    import shutil
    if os.path.exists('outputs'):
        if os.path.exists('CALF_outputs'):
            shutil.rmtree('CALF_outputs')
        os.rename('outputs', 'CALF_outputs')

    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"tight Avg-mAP:  {a_mAP * 100:.2f}%")
    print(f"Avg-mAP:        {a_mAP_visible * 100:.2f}%")
    print()
    print(f"visible tight:  {a_mAP * 100:.2f}%")
    print(f"visible loose:  {a_mAP_visible * 100:.2f}%")
    print()
    print(f"unshown:        {a_mAP_unshown * 100:.2f}%")
    print("=" * 80)

    print("\nCALF reported results:")
    print("  tight:  14.10%")
    print("  loose:  41.61%")
    print("=" * 80)

    results_text = f"""CALF Test Results
================================================================================

tight Avg-mAP:  {a_mAP * 100:.2f}%
Avg-mAP:        {a_mAP_visible * 100:.2f}%

visible tight:  {a_mAP * 100:.2f}%
visible loose:  {a_mAP_visible * 100:.2f}%

unshown:        {a_mAP_unshown * 100:.2f}%

CALF reported results:
  tight:  14.10%
  loose:  41.61%
================================================================================
"""

    with open('CALF_outputs/results.txt', 'w') as f:
        f.write(results_text)

    logging.info("done, predictions saved to /home/o_a38510/sn-spotting/CALF_outputs/")

    return {
        'a_mAP': a_mAP,
        'a_mAP_per_class': a_mAP_per_class,
        'a_mAP_visible': a_mAP_visible,
        'a_mAP_per_class_visible': a_mAP_per_class_visible,
        'a_mAP_unshown': a_mAP_unshown,
        'a_mAP_per_class_unshown': a_mAP_per_class_unshown,
    }


if __name__ == '__main__':
    parser = ArgumentParser(
        description='test CALF on soccernet',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--SoccerNet_path',
        required=True,
        type=str,
        help='path to soccernet data'
    )

    parser.add_argument(
        '--features',
        required=False,
        type=str,
        default="ResNET_TF2_PCA512.npy",
        help='feature filename'
    )

    parser.add_argument(
        '--num_features',
        required=False,
        type=int,
        default=512,
        help='num features'
    )

    parser.add_argument(
        '--model_name',
        required=False,
        type=str,
        default="CALF_benchmark",
        help='model name'
    )

    parser.add_argument(
        '--framerate',
        required=False,
        type=int,
        default=2,
        help='framerate'
    )

    parser.add_argument(
        '--chunk_size',
        required=False,
        type=int,
        default=120,
        help='chunk size'
    )

    parser.add_argument(
        '--receptive_field',
        required=False,
        type=int,
        default=40,
        help='receptive field'
    )

    parser.add_argument(
        '--dim_capsule',
        required=False,
        type=int,
        default=16,
        help='capsule dim'
    )

    args = parser.parse_args()
    main(args)

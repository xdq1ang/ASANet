import argparse


METHOD = 'buildings_WHDLD_Transforms_ppm'
BACKBONE = 'unet'   # resnet
SRC_BATCH_SIZE = [32,1]
TAR_BATCH_SIZE = [17,1]
ITER_SIZE = 1
NUM_WORKERS = 8
DATA_LIST_PATH = [r"datasets\buildings\train_list.txt",r"datasets\buildings\val_list.txt"]
PSEUDO_ROOT = 'results/cityscapes_pseudo_CE80000'
IGNORE_LABEL = 255
INPUT_SIZE = '128,128'
DATA_LIST_PATH_TARGET = [r"datasets\WHDLD\train_list.txt",r"datasets\WHDLD\val_list.txt"]
INPUT_SIZE_TARGET = '128,128'
LEARNING_RATE = 0.00001
LEARNING_RATE_D = 0.00001

MOMENTUM = 0.9
NUM_CLASSES = 2
EPOCH = 100
NUM_STEPS_STOP = 20000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = r'datasets\buildings\训练日志\UNet\model\model.pt'
RESUME = r'datasets\buildings\训练日志\UNet\model\model.pt'
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 1
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
USE_WEIGHT = True
USE_SWA = True
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 1e-3
LAMBDA_ADV_TARGET2 = 2e-4
TARGET = 'buildings'
SET = 'train'
TEMPERATURE = 1.0


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--method", type=str, default=METHOD,
                        help="method name")
    parser.add_argument("--backbone", type=str, default=BACKBONE,
                        help="method name")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--src-batch-size", type=int, default=SRC_BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--tar-batch-size", type=int, default=TAR_BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    # parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
    #                     help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--pseudo-root", type=str, default=PSEUDO_ROOT,
                        help="Path to the directory containing the pseduo label.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    # parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
    #                     help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--use-weight", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--use-swa", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--EPOCH", type=int, default=EPOCH,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--resume", type=str, default=RESUME,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="Which temperature to use.")
    parser.add_argument("--color-dict", type=int, default= {  0 : [0, 0, 0] ,
                                                              1 : [255, 0, 0] ,
                                                              },help="Which temperature to use.")
    return parser.parse_args()

from torch.utils import data, model_zoo
from datasets.gta5_dataset import GTA5DataSet
from utils.boundary_loss import SurfaceLoss,dist_map_transform
from options import gta5asa_opt
args = gta5asa_opt.get_arguments()


if __name__ == "__main__":
    b_loss = SurfaceLoss(idc=[1])
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)
    trainloader = data.DataLoader(GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                    img_size=input_size),
                    batch_size=args.batch_size, 
                    shuffle=False, 
                    num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)
    _, batch = next(trainloader_iter)
    src_img, labels, _, _ = batch
    print(labels.shape)

    disttransform = dist_map_transform([1, 1], 11)
    s = labels.squeeze(dim = 0)
    dist_map_tensor = disttransform(labels)

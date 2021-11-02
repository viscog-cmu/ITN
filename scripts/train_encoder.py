    
import torchsummary
import argparse
import sys

sys.path.append('.')
from topographic.encoder import ITNEncoder

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--imdim', type=int, default=112)
parser.add_argument('--trainset', type=str, default='objfacescenes')
parser.add_argument('--batchsize', type=int, default=512)
parser.add_argument('--no-stride', action='store_true')
parser.add_argument('--polar', action='store_true')
args = parser.parse_args()

stride_tag = '_ns-1' if args.no_stride else ''
polar_tag = '_po-1' if args.polar else ''
model = ITNEncoder(
    encoder_id=args.arch, 
    im_dim=args.imdim,
    dataset_name=args.trainset,
    base_fn=f'enc-{args.arch}{stride_tag}_imd-{args.imdim}{polar_tag}_tr-{args.trainset}',
    batch_size=args.batchsize,
    no_stride=args.no_stride,
    patience=1,
    polar=args.polar,
)    
model.get_tensorboard_writer()

print(torchsummary.summary(model, (3, args.imdim, args.imdim), batch_size=args.batchsize))

model.pretrain(n_epochs=100)
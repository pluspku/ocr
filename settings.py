'''
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--date', help='facades', default = checksum())
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--glr', type=float, default=0.002, help='Generator learning Rate. Default=0.002')
parser.add_argument('--dlr', type=float, default=0.002, help='Discriminator learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?', default = True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
opt = parser.parse_args()
print(opt)
'''

from shared import checksum

cuda = True
date = checksum()
seed = 123
threads = 4

# data params
input_nc = 1
output_nc = 1

# structure params
batch_mode = 'batch'
G_blocks = 9
D_layers = 3
ngf = 64
ndf = 64

# train params
batchSize = 1
testBatchSize = 1
nEpochs = 20000
glr = 0.002
dlr = 0.002
beta1 = 0.5
lamb = 10

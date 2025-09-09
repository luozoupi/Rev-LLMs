import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
# from models import vgg
# from models import vgg_org
import pandas as pd

global count_a
global count_b

count_a = 1
count_b= 1

""" Weight quantization functions """

class _W_tern_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight_mask):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(weight_mask)

        with torch.no_grad():
            ctx.th = 0.05 * input.abs().max()

        # ctx.scale = input[input.ge(ctx.th) + input.le(-ctx.th)].abs().mean() # quantize weight to normal quantized number
        ctx.scale = 1 # quantize weight to actual -1, 0, 1

        output = input.clone().zero_()
        output[input.gt(ctx.th)] = ctx.scale
        output[input.lt(-ctx.th)] = -ctx.scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight_mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[weight_mask] = 0

        return grad_input, None


class _W_quantize_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight_mask, step_size):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(weight_mask)
        ctx.step_size = step_size
        return torch.round(input/ctx.step_size)*ctx.step_size

    @staticmethod
    def backward(ctx, grad_output):
        weight_mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[weight_mask] = 0

        return grad_input, None, None

W_Quantize = _W_quantize_func.apply
W_Quantize_ter = _W_tern_func.apply



""" Activation quantization functions """

def A_Binarize(tensor, quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def A_Quantize(tensor, numBits=8):
    MAX = math.floor(torch.max(tensor).cpu().detach().numpy())
    if MAX > 0:
        div = math.floor(math.log(MAX, 2)) + 1
    else:
        div = 0
    Q_number = numBits - div
    if Q_number > numBits:
        Q_number = numBits
        print('required quantization unit is smaller than what hardware can suppport', "fractional bits requaired:", Q_number, "hardware bits:", numBits)
    if Q_number < 0:
        Q_number = 0
        print('tensor is too large, overflow')
    tensor = tensor.mul(2**Q_number).round().div(2**Q_number)
    threshold = 2 ** numBits - 1  # 边界问题如果0应该存在，-1改成-2
    tensor[tensor > threshold] = threshold
    # print(MAX, div)
    return Q_number, tensor

def A_Quantize_same(tensor, Q_number, numBits=11):
    # MAX = math.floor(torch.max(tensor).cpu().detach().numpy())
    # if MAX > 0:
    #     div = math.floor(math.log(MAX, 2)) + 1
    # else:
    #     div = 0
    # if div < (inputbit - Q_number):
    #     print(Q_number, 'bits utilization is smaller than input size, hardware utilization loss caused, Q get larger')
    # elif div > (inputbit - Q_number):
    #     print(Q_number, 'Q get smaller')
    # else:
    #     print(Q_number, 'Q not change')
    threshold = 2**(numBits - Q_number) - 2 ** (-Q_number)    # 边界问题如果0应该存在，最后(-Q_number)内补一个-1
    tensor[tensor>threshold] = threshold
    tensor = tensor.mul(2**Q_number).round().div(2**Q_number)
    return tensor


class quan_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, quant_type=None, w_bits=None, a_bits=None):
        super(quan_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation,
                                          groups=groups, bias=bias)
        # self.Relu = nn.ReLU(True)
        self.N_bits = w_bits
        self.full_lvls = 2 ** self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        self.get_weight_mask()


        self.a_bits = a_bits
        self.quant_type = quant_type

    def forward(self, input):
        if self.quant_type == "wa":
            # if input.size(1) != 3:
            #     Q_number, input.data = A_Quantize(input.data, numBits=self.a_bits)
            # else:
            #     Q_number = 7  # 输入认为是-1到1， 这种情况下，第一层输入如果为8位，应当是Q7  （第一位符号位，剩余7位小数，此时1与-1 取不到）
            Q_number, input.data = A_Quantize(input.data, numBits=self.a_bits)
            with torch.no_grad():
                step_size = self.weight.abs().max() / self.half_lvls
            # self.weight.org = self.weight.data.clone()
            # self.weight.data = quantize(self.weight.org, self.weight_mask, step_size)
            if self.N_bits == 2: # ternary
                tensor_middle = F.conv2d(input, W_Quantize_ter(self.weight, self.weight_mask), self.bias,
                                         self.stride, self.padding, self.dilation, self.groups)
            else: # multi-bit-length
                tensor_middle = F.conv2d(input, W_Quantize(self.weight, self.weight_mask, step_size), self.bias,
                                         self.stride, self.padding, self.dilation, self.groups)
            tensor_middle = A_Quantize_same(tensor_middle, Q_number=Q_number, numBits = self.a_bits)
            tensor_middle2 = F.relu(tensor_middle)
            #output_data = A_Quantize(tensor_middle2, numBits=self.a_bits)
            output_data = A_Quantize_same(tensor_middle2, Q_number=Q_number, numBits=self.a_bits)

            print('The quantization Q number is:', Q_number)

            global count_a
            global count_b

            in_data = torch.squeeze(input.detach())
            print('size of input data is:', in_data.size())
            q1 = in_data.size()[0]
            in_data = in_data.reshape(q1, -1)
            in_data = in_data.transpose(0, 1)

            in_numpy = in_data.cpu().numpy()
            in_xlsx = pd.DataFrame(in_numpy)

            #out_data = torch.squeeze(output_data.detach())
            out_data = torch.squeeze(output_data.detach())
            q2 = out_data.size()[0]
            out_data = out_data.reshape(q2, -1)

            out_data = out_data.transpose(0, 1)

            out_numpy = out_data.cpu().numpy()
            out_xlsx = pd.DataFrame(out_numpy)

            # count_c = count_a - count_b*13
            import os
            if not os.path.exists('./save_xlsx/'):
                os.makedirs('./save_xlsx/')

            writer = pd.ExcelWriter(
                './save_xlsx/activation_img{0:05d}_conv{1:02d}.xlsx'.format(count_a, count_b))  # å†™å…¥Excelæ‡ä»¶
            in_xlsx.to_excel(writer, 'input', float_format='%.8f')  # â€inputâ€™æ¯å†™å…¥excelçš„sheetå
            out_xlsx.to_excel(writer, 'output', float_format='%.8f')  # â€outputâ€™æ¯å†™å…¥excelçš„sheetå
            writer.save()
            writer.close()

            if not os.path.exists('./save/'):
                os.makedirs('./save/')
            with open('./save/Q_number_img{0:05d}.txt'.format(count_a), "a+") as log_file:
                print(str(Q_number), file=log_file)

            count_b = count_b + 1

            return output_data
        elif self.quant_type == "w":
            with torch.no_grad():
                step_size = self.weight.abs().max()/self.half_lvls
            # self.weight.org = self.weight.data.clone()
            # self.weight.data = quantize(self.weight.org, self.weight_mask, step_size)
            if self.N_bits == 2: # ternary
                return F.conv2d(input, W_Quantize_ter(self.weight, self.weight_mask), self.bias,
                                self.stride, self.padding, self.dilation, self.groups)
            else: # multi-bit-length
                return F.conv2d(input, W_Quantize(self.weight, self.weight_mask, step_size), self.bias,
                                self.stride, self.padding, self.dilation, self.groups)
        elif self.quant_type == "a":
            if input.size(1) != 3:
                Q_number, input.data = A_Quantize(input.data, numBits=self.a_bits)
            else:
                Q_number =7       #  输入认为是-1到1， 这种情况下，第一层输入如果为8位，应当是Q7  （第一位符号位，剩余7位小数，此时1与-1 取不到）
            tensor_middle = F.conv2d(input, self.weight, self.bias,
                                    self.stride, self.padding, self.dilation, self.groups)
            output_data = A_Quantize_same(tensor_middle, Q_number=Q_number)

            print(Q_number)

            # global count_a
            # global count_b

            in_data = torch.squeeze(input.detach())
            print(in_data.size())
            q1 = in_data.size()[0]
            in_data = in_data.reshape(q1, -1)
            in_data = in_data.transpose(0, 1)

            in_numpy = in_data.cpu().numpy()
            in_xlsx = pd.DataFrame(in_numpy)

            out_data = torch.squeeze(output_data.detach())
            q2 = out_data.size()[0]
            out_data = out_data.reshape(q2, -1)

            out_data = out_data.transpose(0, 1)

            out_numpy = out_data.cpu().numpy()
            out_xlsx = pd.DataFrame(out_numpy)

            # count_c = count_a - count_b*13

            writer = pd.ExcelWriter(
                './save_xlsx/activation_img{0:05d}_conv{1:02d}.xlsx'.format(count_a, count_b))  # å†™å…¥Excelæ‡ä»¶
            in_xlsx.to_excel(writer, 'input', float_format='%.8f')  # â€inputâ€™æ¯å†™å…¥excelçš„sheetå
            out_xlsx.to_excel(writer, 'output', float_format='%.8f')  # â€outputâ€™æ¯å†™å…¥excelçš„sheetå
            writer.save()
            writer.close()

            with open('./save/Q_number_img{0:05d}.txt'.format(count_a), "a+") as log_file:
                print(str(Q_number), file=log_file)

            count_b = count_b + 1

            return output_data

    def get_weight_mask(self):
        '''
        This function get the weight mask when load the pretrained model,
        thus all the weights below the preset threshold will be considered
        as the pruned weights, and the returned weight index will be used
        for gradient masking.
        '''

        w_th = 1e-10
        self.weight_mask = self.weight.abs().lt(w_th)

class quanlinear(nn.Linear):

    def __init__(self, in_channel, out_channel, bias=True, w_bits=32, a_bits=None):
        super(quanlinear, self).__init__(in_channel, out_channel, bias=bias)

        self.a_bits = a_bits
        self.w_bits = w_bits # not implement yet
    def forward(self, input):
        _, input.data = A_Quantize(input.data, numBits=self.a_bits)
        # if not hasattr(self.weight,'org'):
        #     self.weight.org=self.weight.data.clone()
        # self.weight.data=Binarize(self.weight.org)


        global count_b
        global count_a
        count_a = count_a + 1
        count_b = 1
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

# def convert_f2q(model_path, w_bits, a_bits):
#     print("\n--------------------------------------------")
#     print("\ninvoke convert_f2q function to convert model")
#     model = vgg.VGG(depth=16, init_weights=True, cfg=None, quant_type="wa", w_bits=w_bits, a_bits=a_bits)
#     model.load_state_dict(torch.load(model_path))
#     model.cuda()
#
#     # """test loader"""
#     # kwargs = {'num_workers': 1, 'pin_memory': True}
#     # test_loader = torch.utils.data.DataLoader(
#     #     datasets.CIFAR10('./datasets/CIFAR10', train=False, transform=transforms.Compose([
#     #         transforms.ToTensor(),
#     #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#     #     ])),
#     #     batch_size=64, shuffle=True, **kwargs)
#     # criterion = torch.nn.CrossEntropyLoss().cuda()
#     # _, acc1 = test(model, criterion, test_loader)
#     # print("before conversion, the simulated model accuracy is: {:.2f}\n".format(acc1))
#
#     full_lvls = 2 ** w_bits
#     half_lvls = (full_lvls - 2) / 2
#
#     w_th = 1e-10
#     for name, weight in model.named_parameters():
#         if len(weight.size()) == 4:
#             # step_size = weight.abs().max() / half_lvls
#             step_size = np.max(np.abs(weight.cpu().detach().numpy())) / half_lvls
#             # weight_mask = weight.abs().lt(w_th)
#             weight_mask = (np.abs(weight.cpu().detach().numpy())) < w_th
#             weight.org = weight.data.clone()
#             if w_bits == 2:
#                 weight.data = W_Quantize_ter(weight.org, weight_mask)
#             else:
#                 weight.data = W_Quantize(weight.org, weight_mask, step_size)
#
#     for name, weight in model.named_parameters():
#         if len(weight.size()) == 4:
#             print("total unique number: [{}] ---> {}".format(np.size(np.unique(weight.cpu().detach().numpy())), np.unique(weight.cpu().detach().numpy())))
#
#     model_name = model_path.split("/")[len(model_path.split("/"))-1]
#     torch.save(model.state_dict(), "./model_quant/unified_{}".format(model_name))
#     # model = vgg_org.VGG(depth=16, init_weights=True, cfg=None)
#     # model.load_state_dict(torch.load("./model_quant/unified_{}".format(model_name)))
#     # model.cuda()
#     # _, acc2 = test(model, criterion, test_loader)
#     # print("\nAFTER conversion, the UNIFIED model accuracy is: {:.2f}\n".format(acc2))



def test(model, criterion, test_loader):
    model.eval()
    losses = AverageMeter()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            losses.update(loss.item(), data.size(0))
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set loss: {:.4f},  * Acc@1: {}/{} ({:.2f}%)\n'.format(
        losses.avg, correct, len(test_loader.dataset),
        100. * float(correct) / float(len(test_loader.dataset))))
    return losses.avg, (100. * float(correct) / float(len(test_loader.dataset)))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
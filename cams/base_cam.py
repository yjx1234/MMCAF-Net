import torch
import torch.nn.functional as F


class BaseCAM(object):
    """Base class for generating CAMs.

    Adapted from: https://github.com/kazuto1011/grad-cam-pytorch
    """
    def __init__(self, model, device, is_binary, is_3d):
        super(BaseCAM, self).__init__()
        self.device = device
        self.is_binary = is_binary
        self.is_3d = is_3d
        self.model = model
        self.model.eval()
        self.inputs = None

    def _encode_one_hot(self, idx):
        #raise ValueError("self.preds是{}".format(self.preds))
        #num_classes = self.preds.size()[-1]  #改：原
        num_classes = self.preds.size()[-1] #改：原，num_classes = self.preds[1].size()[-1]
        one_hot = F.one_hot(torch.tensor([idx]), num_classes=num_classes).float()
        one_hot = one_hot.to(self.device)
        one_hot.requires_grad = True
        return one_hot

    def forward(self, x,tab):
        #self.inputs = x.to(self.device)
        self.inputs = x.to('cuda')   #改：增
        tab = tab.to('cuda')
        self.model.zero_grad()
        #用于清除模型参数的梯度 使得计算模型梯度时不会受到之前梯度的影响
        self.preds = self.model(self.inputs,tab)  #改：用我们的模型时两个参数，训练时为True,其他时候为False

        if self.is_binary:
            #raise ValueError("pred是{}".format(self.preds))
            #self.probs = F.sigmoid(self.preds)[0]  #改：原来的
            self.probs = F.sigmoid(self.preds)[0]  #改：原来self.probs = F.sigmoid(self.preds[1])[0]
        else:
            self.probs = F.softmax(self.preds, dim=1)[0]
        self.prob, self.idx = self.probs.sort(0, True)#对第0轴降序排序

        return self.prob, self.idx

    def backward(self, idx):
        #self.model.zero_grad()   #增
        one_hot = self._encode_one_hot(idx)
        #raise ValueError("one_hot的形状是{},one_hot是{}".format(one_hot.shape, one_hot))
        #self.preds.backward(gradient=one_hot, retain_graph=True)  #改：原来的
        #raise ValueError("preds[1]的形状是{},pred[1]是{}".format(self.preds[1].shape, self.preds[1]))
        self.preds.backward(gradient=one_hot, retain_graph=True)  ##改：原self.preds[1].backward(gradient=one_hot, retain_graph=True)

    def get_cam(self, target_layer):
        raise NotImplementedError

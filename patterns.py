import torch

#(1, 4, 96, 96)
def Stacked_Img_Pattern(img_size, trigger_size, loc = (0,0), min = -255, max = 255, checker = True):
    pattern = torch.zeros(size = img_size)
    for i in range(trigger_size[0]):
        for j in range(trigger_size[1]):
            if checker and (i+j)%2==0:
                pattern[:, :, i + loc[0],j + loc[1]] = min
            else:
                pattern[:, :, i+loc[0],j+loc[1]] = max
    return pattern.long()

def Single_Stacked_Img_Pattern(img_size, trigger_size, loc = (0,0), min = -255, max = 255, checker = True):
    pattern = torch.zeros(size = img_size)
    for i in range(trigger_size[0]):
        for j in range(trigger_size[1]):
            if checker and (i+j)%2==0:
                pattern[:, i + loc[0],j + loc[1]] = min
            else:
                pattern[:, i+loc[0],j+loc[1]] = max
    return pattern.long()
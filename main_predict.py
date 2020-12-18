import cv2
import torch
import matplotlib.pyplot as plt

from model_cnn import *


def prediction(img, model_cnn, model_state, device='cpu'):
    # image processing
    img_print = cv2.resize(cv2.imread(img_path)[:, :, ::-1], (600, 600))
    img_resize = cv2.resize(img, (32, 32))
    img_tensor = torch.from_numpy(img_resize).view(1, 1, 32, 32).type('torch.FloatTensor')

    # model
    device = torch.device(device)
    model = model_cnn
    model.to(device)
    state_dict = torch.load(model_state)
    model.load_state_dict(state_dict)

    # predict
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        output_sof = torch.softmax(output, dim=1)
        prob, pred = torch.max(output_sof, 1)
        prob, pred = prob.item(), pred.item()

    cv2.putText(img_print, str(pred) + ' (' + str(prob) + ')', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0 , 255), 2)
    return pred, img_print

img_path = 'img/3.jpg'
img = cv2.imread(img_path, 0)
pred, img = prediction(img, Classifier(), 'classifier_digit2.pt')
plt.figure(), plt.imshow(img)

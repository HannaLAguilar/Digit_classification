import cv2
import torch
import matplotlib.pyplot as plt

from model_cnn import Classifier


def prediction(img_path, model_cnn, model_state, device='cpu'):
    # image processing
    img = cv2.imread(img_path, 0)
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
        _, pred = torch.max(output, 1)
    pred = pred.item()

    cv2.putText(img_print, 'Prediction: ' + str(pred), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    return pred, img_print


img_path = 'img/3.jpg'
pred, img = prediction(img_path, Classifier(), 'classifier_digit.pt')
plt.figure(), plt.imshow(img)
print(pred)

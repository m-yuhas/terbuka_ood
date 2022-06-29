import json
import os

import cv2
import numpy
import torch
import torchvision

from vae import Vae

weights = 'raw_mse_50.pt'
test_images = 'confetti'

model = torch.load(weights)
model.eval()
model.batch = 1

kl_cal = None
mse_cal = None
with open(f'cal_{weights.replace(".pt", "")}.json', 'r') as cal_f:
    data = json.loads(cal_f.read())
    kl_cal = numpy.array(data['kl_loss'])
    mse_cal = numpy.array(data['mse_loss'])

writer = cv2.VideoWriter(
    f'{test_images}.avi',
    cv2.VideoWriter_fourcc(*'XVID'),
    30.0,
    (1280, 480))

ti = os.listdir(test_images)
ti.sort()
for f in ti:
    new_frame = numpy.zeros((480, 1280, 3), dtype=numpy.uint8)

    img = cv2.imread(os.path.join(test_images, f))
    new_frame[:, :640, :] = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = torchvision.transforms.functional.to_tensor(img)
    x = torchvision.transforms.functional.resize(x, (120, 160))
    x = torchvision.transforms.functional.rgb_to_grayscale(x)
    x = x.unsqueeze(0).to('cuda')
    out, mu, logvar = model(x)
    x_hat = out.clone()

    out = out.detach().cpu().numpy()
    out = numpy.squeeze(out)
    out = cv2.normalize(out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out = cv2.resize(out, (640, 480))
    new_frame[:, 640:, :] = out

    cv2.putText(new_frame, 'KL Score', (1050, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.rectangle(new_frame, pt1=(1050, 30), pt2=(1250, 55), color=(0, 255, 255), thickness=2)
    cv2.putText(new_frame, 'Reconstruction Loss', (1050, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.rectangle(new_frame, pt1=(1050, 80), pt2=(1250, 105), color=(255, 255, 0), thickness=2)

    kl_loss = torch.mul(
        input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
        other=0.5)
    #ce_loss = torch.nn.functional.binary_cross_entropy(
    #    input=x_hat,
    #    target=x.unsqueeze(0).to('cuda'),
    #    reduction='sum')
    mse_loss = torch.sum((x - x_hat).pow(2))

    #print(ce_loss)
    kl_loss = kl_loss.detach().cpu().numpy()
    mse_loss = mse_loss.detach().cpu().numpy()
    p_kl = numpy.count_nonzero(kl_cal < kl_loss) / kl_cal.size
    p_mse = numpy.count_nonzero(mse_cal < mse_loss) / mse_cal.size

    cv2.rectangle(
        new_frame,
        pt1=(1050, 30),
        pt2=(int(p_kl * 200 + 1050), 55),
        color=(0, 255, 255),
        thickness=-1)
    cv2.rectangle(
        new_frame,
        pt1=(1050, 80),
        pt2=(int(p_mse * 200 + 1050), 105),
        color=(255, 255, 0),
        thickness=-1)



    writer.write(new_frame)

writer.release()

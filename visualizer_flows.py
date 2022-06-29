import json
import os

import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy
import torch
import torchvision

from vae import Vae

#matplotlib.use('GTK3Agg')
plt.rc('xtick', labelsize=1)
plt.rc('ytick', labelsize=1)

weights_h = 'horiz_flow_6_200.pt'
weights_v = 'vert_flow_6_200.pt'
test_images = 'fog'

model_h = torch.load(weights_h)
model_h.eval()
model_h.batch = 1

model_v = torch.load(weights_v)
model_v.eval()
model_v.batch = 1

kl_cal_h = None
mse_cal_h = None
with open(f'cal_{weights_h.replace(".pt", "")}.json', 'r') as cal_f:
    data = json.loads(cal_f.read())
    kl_cal_h = numpy.array(data['kl_loss'])
    mse_cal_h = numpy.array(data['mse_loss'])

kl_cal_v = None
mse_cal_v = None
with open(f'cal_{weights_v.replace(".pt", "")}.json', 'r') as cal_f:
    data = json.loads(cal_f.read())
    kl_cal_v = numpy.array(data['kl_loss'])
    mes_cal_v = numpy.array(data['mse_loss'])

writer = cv2.VideoWriter(
    f'{test_images}_optflow.avi',
    cv2.VideoWriter_fourcc(*'XVID'),
    30.0,
    (1280, 480))

ti = os.listdir(test_images)
ti.sort()

flow = None
last_frame = None
h_buf = [None] * 6
v_buf = [None] * 6
ptr = 0
count = 0

for f in ti:
    new_frame = numpy.zeros((480, 1280, 3), dtype=numpy.uint8)

    img = cv2.imread(os.path.join(test_images, f))
    new_frame[:, :640, :] = img

    img = cv2.resize(img, (160, 120))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if last_frame is not None:
        flow = cv2.calcOpticalFlowFarneback(
            last_frame,
            img,
            flow,
            pyr_scale=0.5,
            levels=1,
            iterations=1,
            winsize=15,
            poly_n=5,
            poly_sigma=1.1,
            flags=0 if flow is None else cv2.OPTFLOW_USE_INITIAL_FLOW)
        h_buf[ptr] = numpy.copy(flow[:, :, 0])
        v_buf[ptr] = numpy.copy(flow[:, :, 1])
        if count >= 5:
            horiz = numpy.zeros((120, 160, 6))
            vert = numpy.zeros((120, 160, 6))
            ptr = (ptr + 1) % 6
            for i in range(6):
                horiz[:, :, i] = h_buf[ptr]
                vert[:, :, i] = v_buf[ptr]
                ptr = (ptr + 1) % 6
            
            sample_h = torch.from_numpy(horiz)
            sample_v = torch.from_numpy(vert)
            sample_h = torch.swapaxes(sample_h, 1, 2)
            sample_v = torch.swapaxes(sample_v, 1, 2)
            sample_h = torch.swapaxes(sample_h, 0, 1)
            sample_v = torch.swapaxes(sample_v, 0, 1)

            sample_h = sample_h.nan_to_num(0)
            sample_v = sample_v.nan_to_num(0)
            sample_h = ((sample_h + 64) / 128).clamp(0, 1).type(torch.FloatTensor)
            sample_v = ((sample_v + 64) / 128).clamp(0, 1).type(torch.FloatTensor)
            
            x_hat_h, mu_h, logvar_h = model_h(sample_h.unsqueeze(0).to('cuda'))
            y_hat_v, mu_v, logvar_v = model_v(sample_v.unsqueeze(0).to('cuda'))

            kl_h = torch.mul(
                input=torch.sum(mu_h.pow(2) + logvar_h.exp() - logvar_h - 1),
                other=0.5)
            kl_v = torch.mul(
                input=torch.sum(mu_v.pow(2) + logvar_v.exp() - logvar_v - 1),
                other=0.5)

            #h_img = cv2.normalize(flow[:, :, 0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #v_img = cv2.normalize(flow[:, :, 0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #h_img = cv2.cvtColor(h_img, cv2.COLOR_GRAY2BGR)
            #v_img = cv2.cvtColor(v_img, cv2.COLOR_GRAY2BGR)
            #h_img = cv2.resize(h_img, (320, 240))
            #v_img = cv2.resize(v_img, (320, 240))

            #new_frame[:240, 640:960, :] = h_img
            #new_frame[240:, 640:960, :] = v_img

            h, w = 120, 160
            of_x, of_y = flow[..., 0], flow[..., 1]
            U = cv2.resize(of_x, None, fx=0.2, fy=0.2)
            V = cv2.resize(of_y, None, fx=0.2, fy=0.2)
            M = numpy.sqrt(numpy.add(U ** 2, V ** 2))
            X = numpy.arange(0, w, 5)
            Y = numpy.arange(0, h, 5)
            fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
            ax = fig.add_subplot(111)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            q = ax.quiver(X, Y, U, V, M, cmap=plt.cm.jet)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = numpy.frombuffer(fig.canvas.tostring_rgb(), dtype=numpy.uint8)
            buf.shape = (h, w, 3)
            buf = buf[:]
            buf = numpy.flip(buf, axis=0)
            buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            buf = cv2.resize(buf, (480, 240))
            plt.close(fig)
            new_frame[120:360, 760:1240, :] = buf

            #cv2.putText(new_frame, 'Horizontal Flows', (650, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            #cv2.putText(new_frame, 'Vertical Flows', (650, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            cv2.putText(new_frame, 'OODScore Horizontal', (1050, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.rectangle(new_frame, pt1=(1050, 30), pt2=(1250, 55), color=(0, 255, 255), thickness=2)
            cv2.putText(new_frame, 'OODScore Vertical', (1050, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 255, 0), thickness=2)
            cv2.rectangle(new_frame, pt1=(1050, 80), pt2=(1250, 105), color=(255, 255, 0), thickness=2)
            kl_h = kl_h.detach().cpu().numpy()
            kl_v = kl_v.detach().cpu().numpy()
            p_kl_h = numpy.count_nonzero(kl_cal_h < kl_h) / kl_cal_h.size
            p_kl_v = numpy.count_nonzero(kl_cal_v < kl_v) / kl_cal_v.size
            
            cv2.rectangle(
                new_frame,
                pt1=(1050, 30),
                pt2=(int(p_kl_h * 200 + 1050), 55),
                color=(0, 255, 255),
                thickness=-1)
            cv2.rectangle(
                new_frame,
                pt1=(1050, 80),
                pt2=(int(p_kl_v * 200 + 1050), 105),
                color=(255, 255, 0),
                thickness=-1)

            writer.write(new_frame)


    last_frame = numpy.copy(img)
    count += 1

writer.release()

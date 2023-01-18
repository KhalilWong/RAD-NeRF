import numpy as np
import os
import cv2

def showlms(root_path):
    if not os.path.exists(os.path.join(root_path, 'ori_imgs', 'LMSFRAMES')):
        os.mkdir(os.path.join(root_path, 'ori_imgs', 'LMSFRAMES'))
    for file in os.listdir(os.path.join(root_path, 'ori_imgs')):
        if file.split('.')[-1] == 'lms':
            lms = np.loadtxt(os.path.join(root_path, 'ori_imgs', file))
            img = cv2.imread(os.path.join(root_path, 'ori_imgs', file.replace('lms', 'jpg')))

            f_xmin, f_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max())
            f_ymin, f_ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            img = cv2.rectangle(img, (f_ymin, f_xmin), (f_ymax, f_xmax), (0, 0, 255), thickness = 1)

            lips = slice(48, 60)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            l = max(xmax - xmin, ymax - ymin) // 2
            l_xmin = max(0, cx - l)
            l_xmax = min(512, cx + l)
            l_ymin = max(0, cy - l)
            l_ymax = min(512, cy + l)
            img = cv2.rectangle(img, (l_ymin, l_xmin), (l_ymax, l_xmax), (0, 255, 0), thickness = 1)
            #print(file.replace('lms', 'jpg'))
            cv2.imwrite(os.path.join(root_path, 'ori_imgs', 'LMSFRAMES', file.replace('lms', 'jpg')), img)

if __name__ == '__main__':
    showlms('data/shiying')

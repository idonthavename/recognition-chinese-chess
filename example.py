import cv2 as cv
import numpy as np
import os
from scipy import misc, ndimage
import sys

# 获取象棋的棋盘坐标


def get_cell(input_path):
    img = cv.imread(input_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    gradX = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, rectKernel)
    ret1, th1 = cv.threshold(gradX, 35, 255, cv.THRESH_BINARY_INV)
    edges = cv.Canny(th1, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi/180, 143)

    zx = []
    sx = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        if (x1 > 10 and x2 > 10 and abs(x1 - x2) < 10):
            # print("x", (x1, x2, y1, y2))
            sx.append((x1, x2, y1, y2))
        elif (y1 > 10 and y2 > 10 and abs(y1 - y2) < 10):
            # print("y", (x1, x2, y1, y2))
            zx.append((x1, x2, y1, y2))
        # cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    sx = sorted(sx, key=lambda x: x[0])
    zx = sorted(zx, key=lambda x: x[2])

    sxFilter = []
    if len(sx) > 0:
        sxFilter.append(sx[0])
        tempsx = sx[0]
        for i in range(len(sx))[1:]:
            # print(sx[i][0],tempsx[0])
            if (sx[i][0] - tempsx[0]) > 20:
                tempsx = sx[i]
                sxFilter.append(sx[i])

    zxFilter = []
    if len(zx) > 0:
        zxFilter.append(zx[0])
        tempzx = zx[0]
        for i in range(len(zx))[1:]:
            # print(zx[i][0],tempzx[0])
            if (zx[i][2] - tempzx[2]) > 20:
                tempzx = zx[i]
                zxFilter.append(zx[i])
    return sxFilter, zxFilter


# 获取 棋盘中棋子的位置信息
def get_chess_position(img_path):
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (13, 13))
    gradX = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, rectKernel)
    questionCnts = []
    ret1, th1 = cv.threshold(gradX, 170, 255, cv.THRESH_BINARY)
    grad2 = cv.morphologyEx(th1, cv.MORPH_DILATE, rectKernel)
    grad2 = cv.morphologyEx(grad2, cv.MORPH_DILATE, rectKernel)

    # 找到每一个圆圈轮廓
    cnts = cv.findContours(grad2, cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)[0]

    for i in range(len(cnts)):
        x, y, w, h = cv.boundingRect(cnts[i])
        print(x, y, w, h)
        if (abs(w - h) > 20):
            continue
        # questionCnts.append(cnts[i])
        questionCnts.append((x, y, w, h))
    return questionCnts


# 棋子的匹配加载
path = '/Users/apple/Desktop/python/recognition-chinese-chess/Dataset/train/'
name_dict = {"b_c.png": "黑方车", "b_j.png": "黑方将", "b_m.png": "黑方马", "b_p.png": "黑方炮", "b_s.png": "黑方士", "b_x.png": "黑方象", "b_z.png": "黑方卒",
             "r_c.png": "红方车", "r_j.png": "红方帅", "r_m.png": "红方马", "r_p.png": "红方炮", "r_s.png": "红方士", "r_x.png": "红方象", "r_z.png": "红方兵"}

if os.path.exists(path):
    files = os.listdir(path)
    print(files)
    sys.exit()


def get_match(old_img):
    if len(files) > 0:
        # print(files)
        check_score = 0
        check_img = ""
        for fileName in files:
            tempImg = cv.imread(path + fileName)

            '''
            ①TM_SQDIFF是平方差匹配；TM_SQDIFF_NORMED是标准平方差匹配。利用平方差来进行匹配, 最好匹配为0.匹配越差, 匹配值越大。
            ②TM_CCORR是相关性匹配；TM_CCORR_NORMED是标准相关性匹配。采用模板和图像间的乘法操作, 数越大表示匹配程度较高, 0
            表示最坏的匹配效果。
            ③TM_CCOEFF是相关性系数匹配；TM_CCOEFF_NORMED是标准相关性系数匹配。将模版对其均值的相对值与图像对其均值的相关值进行匹配, 1
            表示完美匹配, -1
            表示糟糕的匹配, 0
            表示没有任何相关性(随机序列)。

            总结：随着从简单的测量(平方差)
            到更复杂的测量(相关系数), 我们可获得越来越准确的匹配(同时也意味着越来越大的计算代价)。
            '''
            result = cv.matchTemplate(cv.resize(
                old_img, None, fx=0.8, fy=0.8, interpolation=cv.INTER_CUBIC), tempImg, cv.TM_CCOEFF_NORMED)
            # print(result)
            (_, score, _, _) = cv.minMaxLoc(result)
            if check_score == 0 or score > check_score:
                check_score = score
                check_img = fileName
            # print(cv.minMaxLoc(result),name_dict[check_img])
        return name_dict[check_img]
    else:
        print('this path not exist')
        return None


input_path = './4.jpg'
sxFilter, zxFilter = get_cell(input_path)
questionCnts = get_chess_position(input_path)


img = cv.imread(input_path)
img_show = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

chess_array = [[] for i in range(10)]
if(len(sxFilter) == 9 and len(zxFilter) == 10):
    # 竖线 直线
    print("---------------------")
    for i in range(len(zxFilter)):
        for j in range(len(sxFilter)):
            x1, x2, y1, y2 = zxFilter[i]
            x3, x4, y3, y4 = sxFilter[j]
            check = False
            print(zxFilter[i], "-", sxFilter[j], i, j)
            for k in range(len(questionCnts)):
                x, y, w, h = questionCnts[k]
                # 获取中心坐标
                centre_x = x+w/2
                centre_y = y+h/2
                # 范围之内匹配成功
                print(centre_x, "-", centre_y)
                if(abs(centre_x-x3) < 25 and abs(centre_y-y1) < 25):
                    print("命中!")
                    check = True
                    im = img[y:y + h, x:x + w]
                    orc = get_match(im)
                    if orc is not None:
                        chess_array[i].append(orc)
                    else:
                        print("未识别...")
            if not check:
                chess_array[i].append("空空空")
else:
    print("识别出错了!")

for i in range(10):
    for j in range(9):
        print(chess_array[i][j], end="\t")
    print()


for i in range(len(sxFilter)):
    line = sxFilter[i]
    cv.line(img, (line[0], line[2]), (line[1], line[3]), (0, 0, 255), 2)
    # print(line)
    cv.putText(img, str(i), (line[0],  500),
               cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

for i in range(len(zxFilter)):
    line = zxFilter[i]
    cv.line(img, (line[0], line[2]), (line[1], line[3]), (0, 0, 255), 2)
    # print(line)
    cv.putText(img, str(i), (300,  line[2]),
               cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

res = np.hstack((img_show, img))
showIMG = cv.resize(res, None, fx=0.7, fy=0.6, interpolation=cv.INTER_CUBIC)
cv.imshow("TEST", showIMG)

k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
cv.destroyAllWindows()

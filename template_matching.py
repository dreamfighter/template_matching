import cv2
import numpy as np
import sys
import argparse
import imutils
import math
from matplotlib import pyplot as plt
from skimage import exposure

def distance(p1,p2):  
	dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)  
	return dist

def matchTemplate(imgName):
	img_rgb = cv2.imread(imgName)
	img_result = cv2.imread(imgName)
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	template = cv2.imread('mark.png',0)

	w, h = template.shape[::-1]
	height = img_rgb.shape[0]
	width = img_rgb.shape[1]
	print("image size = {},{}".format(width,height))

	res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
	threshold = 0.45
	loc = np.where(res >= threshold)
	i = 0 
	f = set()
	xmin = 50
	ymin = 50
	prevpt = (0,0)
	result = []
	r = {}
	for pt in zip(*loc[::-1]):
		sensitivity = 100
		sensitivityX = round(pt[0]/sensitivity)
		sensitivityY = round(pt[1]/sensitivity)


		if (sensitivityX, sensitivityY) not in f:
			if sensitivityY not in r:
				r[int(sensitivityY)] = []

			r[int(sensitivityY)].append(pt)
			print(sensitivity)
			result.append(pt)
			f.add((sensitivityX, sensitivityY))
			#cv2.circle(img_rgb, pt, 3, (0, 0, 255), -1)
			#cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

	for listpt in r.values():
		#for l in listpt:
		print(listpt)
	#a = np.array(result)
	#ind = np.lexsort((a[:,0],a[:,1])) 
	#b = a[ind]
	a = np.array(result)
	dt = [('col1', a.dtype),('col2', a.dtype)]
	assert a.flags['C_CONTIGUOUS']
	b = a.ravel().view(dt)
	b.sort(order=['col1','col2'])

	for pt in b:
		i+=1
		imgCopy = img_result.copy()

		nodes = []
		p1 = (pt[0]-5, pt[1]-5)
		if i==1:
			if pt[1]-5>ymin:
				p1 = (xmin - 35, ymin - 25)
			p2 = (p1[0], pt[1]-5)
			p3 = (pt[0]-5, pt[1]-5)
			p4 = (pt[0]-5, pt[1] + h + 5)
			p5 = (width - 5, pt[1] + h + 5)
			p6 = (width - 5, p1[1])
			nodes.append(p1)
			nodes.append(p2)
			nodes.append(p3)
			nodes.append(p4)
			nodes.append(p5)
			nodes.append(p6)
		else:
			p1 = (prevpt[0]-5, prevpt[1]-5)
			if xmin > prevpt[0] - 5:
				p1 = (width - 10, prevpt[1] + h + 10)
			if pt[1]-5 < p1[1] + 50:
				p2 = (pt[0]-5,p1[1])
				p3 = (pt[0]-5, pt[1] + h + 10)
				p4 = (p1[0], pt[1] + h + 10)
			else:
				p2 = (xmin - 35,p1[1])
				p3 = (p2[0], pt[1] - 5)
				p4 = (pt[0] - 5, pt[1] - 5)
			
			if pt[1]-5 > p1[1] + 50:
				p5 = (pt[0] - 5, pt[1] + h + 5)
				p6 = (width - 10, pt[1] + h + 5)
				p7 = (width - 10, p1[1] + h + 10)
				p8 = (p1[0], p1[1] + h + 10)

			nodes.append(p1)
			nodes.append(p2)
			nodes.append(p3)
			nodes.append(p4)
			if pt[1]-5 > p1[1] + 50:
				nodes.append(p5)
				nodes.append(p6)
				nodes.append(p7)
				nodes.append(p8)

		
		
		#nodes = np.array(nodes)
		
		nodes = np.int32([nodes])

		cv2.polylines(imgCopy, nodes, True, (0,0,255), 2, lineType=cv2.LINE_AA)
		prevpt = pt
		#if pt[0]-10<xmin:
		cv2.rectangle(img_rgb, (pt[0],pt[1]), (pt[0] + w, pt[1] + h + 5), (0,0,255), 2)
		cv2.putText(img_rgb,"{}".format(i), (pt[0],pt[1]), cv2.FONT_HERSHEY_DUPLEX, 2, 255)

		cv2.imwrite('result/res-{}.png'.format(i),imgCopy)
		print('{} - ({} {})'.format(i, pt[0], pt[1]))
		if len(sys.argv)>2:
			if "{}".format(i) == sys.argv[2]:
				break;

	cv2.imwrite('res.png'.format(i),img_rgb)
	
	print('found = {}'.format(len(f)))

	#cv2.imwrite('res.png',img_rgb)
	return;

def findContour( imgName ):   
	image = cv2.imread(imgName)
	#ratio = image.shape[0] / 300.0
	#orig = image.copy()
	#image = imutils.resize(image, height = 300)
	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	edged = cv2.Canny(gray, 30, 200)
	
	cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
	screenCnt = None

	# loop over our contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.015 * peri, True)
		# if our approximated contour has four points, then
		# we can assume that we have found our screen
		cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)

		#if len(approx) == 4:
		#	screenCnt = approx
		#	break

	
	#cv2.imshow("Game Boy Screen", image)
	#cv2.waitKey(0)
	cv2.imwrite('res1.png',image)
	return edged;

def findContour2( imgName ):
	canny_img = findContour(imgName)
	im2, contours, hierarchy = cv2.findContours(canny_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	try: hierarchy = hierarchy[0]
	except: hierarchy = []

	height, width, _ = canny_img.shape
	min_x, min_y = width, height
	max_x = max_y = 0

	# computes the bounding box for the contour, and draws it on the frame,
	for contour, hier in zip(contours, hierarchy):
	    (x,y,w,h) = cv2.boundingRect(contour)
	    min_x, max_x = min(x, min_x), max(x+w, max_x)
	    min_y, max_y = min(y, min_y), max(y+h, max_y)
	    if w > 80 and h > 80:
	        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)

	if max_x - min_x > 0 and max_y - min_y > 0:
	    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
	cv2.imwrite('res2.png',frame)
	return;

print(sys.argv[1])

matchTemplate(sys.argv[1])
findContour(sys.argv[1])
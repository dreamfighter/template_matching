import cv2
import numpy as np
import sys
import argparse
import imutils
import math
import sqlite3
from matplotlib import pyplot as plt
from skimage import exposure

conn = sqlite3.connect('block-v2.db')

frame = {1:(150,335,70)}

PAGE_SURA_START = [ 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
			2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
			2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
			3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
			4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
			4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
			5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
			6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
			7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9,
			9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10,
			10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
			11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12,
			12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
			14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
			16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18,
			18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19,
			19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21,
			21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23,
			23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25,
			25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27,
			27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
			29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 31, 31, 31,
			31, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34,
			34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 37, 37,
			37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39,
			39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41,
			41, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 44, 44, 44,
			45, 45, 45, 45, 46, 46, 46, 46, 47, 47, 47, 47, 48, 48, 48, 48, 48,
			49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 53, 53, 53, 54, 54, 54, 55,
			55, 55, 56, 56, 56, 57, 57, 57, 57, 58, 58, 58, 58, 59, 59, 59, 60,
			60, 60, 61, 62, 62, 63, 64, 64, 65, 65, 66, 66, 67, 67, 67, 68, 68,
			69, 69, 70, 70, 71, 72, 72, 73, 73, 74, 74, 75, 76, 76, 77, 78, 78,
			79, 80, 81, 82, 83, 83, 85, 86, 87, 89, 89, 91, 92, 95, 97, 98,
			100, 103, 106, 109, 112 ]

PAGE_AYAH_START = [ 1, 1, 6, 17, 25, 30, 38, 49, 58,
			62, 70, 77, 84, 89, 94, 102, 106, 113, 120, 127, 135, 142, 146,
			154, 164, 170, 177, 182, 187, 191, 197, 203, 211, 216, 220, 225,
			231, 234, 238, 246, 249, 253, 257, 260, 265, 270, 275, 282, 283, 1,
			10, 16, 23, 30, 38, 46, 53, 62, 71, 78, 84, 92, 101, 109, 116, 122,
			133, 141, 149, 154, 158, 166, 174, 181, 187, 195, 1, 7, 12, 15, 20,
			24, 27, 34, 38, 45, 52, 60, 66, 75, 80, 87, 92, 95, 102, 106, 114,
			122, 128, 135, 141, 148, 155, 163, 171, 176, 3, 6, 10, 14, 18, 24,
			32, 37, 42, 46, 51, 58, 65, 71, 77, 83, 90, 96, 104, 109, 114, 1,
			9, 19, 28, 36, 45, 53, 60, 69, 74, 82, 91, 95, 102, 111, 119, 125,
			132, 138, 143, 147, 152, 158, 1, 12, 23, 31, 38, 44, 52, 58, 68,
			74, 82, 88, 96, 105, 121, 131, 138, 144, 150, 156, 160, 164, 171,
			179, 188, 196, 1, 9, 17, 26, 34, 41, 46, 53, 62, 70, 1, 7, 14, 21,
			27, 32, 37, 41, 48, 55, 62, 69, 73, 80, 87, 94, 100, 107, 112, 118,
			123, 1, 7, 15, 21, 26, 34, 43, 54, 62, 71, 79, 89, 98, 107, 6, 13,
			20, 29, 38, 46, 54, 63, 72, 82, 89, 98, 109, 118, 5, 15, 23, 31,
			38, 44, 53, 64, 70, 79, 87, 96, 104, 1, 6, 14, 19, 29, 35, 43, 6,
			11, 19, 25, 34, 43, 1, 16, 32, 52, 71, 91, 7, 15, 27, 35, 43, 55,
			65, 73, 80, 88, 94, 103, 111, 119, 1, 8, 18, 28, 39, 50, 59, 67,
			76, 87, 97, 105, 5, 16, 21, 28, 35, 46, 54, 62, 75, 84, 98, 1, 12,
			26, 39, 52, 65, 77, 96, 13, 38, 52, 65, 77, 88, 99, 114, 126, 1,
			11, 25, 36, 45, 58, 73, 82, 91, 102, 1, 6, 16, 24, 31, 39, 47, 56,
			65, 73, 1, 18, 28, 43, 60, 75, 90, 105, 1, 11, 21, 28, 32, 37, 44,
			54, 59, 62, 3, 12, 21, 33, 44, 56, 68, 1, 20, 40, 61, 84, 112, 137,
			160, 184, 207, 1, 14, 23, 36, 45, 56, 64, 77, 89, 6, 14, 22, 29,
			36, 44, 51, 60, 71, 78, 85, 7, 15, 24, 31, 39, 46, 53, 64, 6, 16,
			25, 33, 42, 51, 1, 12, 20, 29, 1, 12, 21, 1, 7, 16, 23, 31, 36, 44,
			51, 55, 63, 1, 8, 15, 23, 32, 40, 49, 4, 12, 19, 31, 39, 45, 13,
			28, 41, 55, 71, 1, 25, 52, 77, 103, 127, 154, 1, 17, 27, 43, 62,
			84, 6, 11, 22, 32, 41, 48, 57, 68, 75, 8, 17, 26, 34, 41, 50, 59,
			67, 78, 1, 12, 21, 30, 39, 47, 1, 11, 16, 23, 32, 45, 52, 11, 23,
			34, 48, 61, 74, 1, 19, 40, 1, 14, 23, 33, 6, 15, 21, 29, 1, 12, 20,
			30, 1, 10, 16, 24, 29, 5, 12, 1, 16, 36, 7, 31, 52, 15, 32, 1, 27,
			45, 7, 28, 50, 17, 41, 68, 17, 51, 77, 4, 12, 19, 25, 1, 7, 12, 22,
			4, 10, 17, 1, 6, 12, 6, 1, 9, 5, 1, 10, 1, 6, 1, 8, 1, 13, 27, 16,
			43, 9, 35, 11, 40, 11, 1, 14, 1, 20, 18, 48, 20, 6, 26, 20, 1, 31,
			16, 1, 1, 1, 7, 35, 1, 1, 16, 1, 24, 1, 15, 1, 1, 8, 10, 1, 1, 1, 1 ]

SURA_NUM_AYAHS = [ 7, 286, 200, 176, 120, 165, 206, 75,
			129, 109, 123, 111, 43, 52, 99, 128, 111, 110, 98, 135, 112, 78,
			118, 64, 77, 227, 93, 88, 69, 60, 34, 30, 73, 54, 45, 83, 182, 88,
			75, 85, 54, 53, 89, 59, 37, 35, 38, 29, 18, 45, 60, 49, 62, 55, 78,
			96, 29, 22, 24, 13, 14, 11, 11, 18, 12, 12, 30, 52, 52, 44, 28, 28,
			20, 56, 40, 31, 50, 40, 46, 42, 29, 19, 36, 25, 22, 17, 19, 26, 30,
			20, 15, 21, 11, 8, 8, 19, 5, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3, 6,
			3, 5, 4, 5, 6 ]

def createTable():
	
	c = conn.cursor()

	# Create table
	c.execute('''CREATE TABLE block
	             (id integer, x real, y real, ratioWidth integer, ratioHeight integer, block integer, page integer, surah integer, ayah integer, createdOn text, updatedOn text)''')

	# Insert a row of data
	#c.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")

	# Save (commit) the changes
	conn.commit()

	# We can also close the connection if we are done with it.
	# Just be sure any changes have been committed or they will be lost.
	conn.close()

def insert(id,x,y,ratioWidth,ratioHeight,block,page,surah,ayah,createdOn,updatedOn):
	c = conn.cursor()

	# Create table
	c.execute("INSERT INTO block VALUES ({},{},{},{},{},{},{},{},{},{},{})".format(id,x,y,ratioWidth,ratioHeight,block,page,surah,ayah,createdOn,updatedOn))

	# Insert a row of data
	#c.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")

	# Save (commit) the changes
	conn.commit()

	# We can also close the connection if we are done with it.
	# Just be sure any changes have been committed or they will be lost.
	conn.close()

def getAyahCountFromNumberSurah(surah):
	surah = surah - 1
	if surah >= 0 and surah < len(SURA_NUM_AYAHS):
		return SURA_NUM_AYAHS[surah]
	
	return -1
	
def getFirstAyahFromPage(page):
	if page > 0 and page <= len(PAGE_AYAH_START):
		return PAGE_AYAH_START[page - 1]
	
	return -1
	
def getSuraNumberFromPage(page):
	if page > 0 and page <= len(PAGE_SURA_START):
		return PAGE_SURA_START[page - 1]
	
	return None
	
def getListAyahFromPage(page):

	surah = getSuraNumberFromPage(page)
	min = getFirstAyahFromPage(page)
	max = getFirstAyahFromPage(page + 1)
	if max == 1:
		surahNumber = getSuraNumberFromPage(page)
		max = getAyahCountFromNumberSurah(surahNumber)
		max +=1
	
	list = []
	for i in range(min, max):
		ayah = "{}".format(i).ljust(3, '0')
		list.append((page,surah,i))
		#list.append("{}{}{}".format(surah, ayah, ".png"))
	
	return list


def distance(p1,p2):
	dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)  
	return dist

def sort_x(t):
	return t[0]

def matchTemplateHeader(imgName,page):
	img1_rgb = cv2.imread(imgName)
	img1_result = cv2.imread(imgName)
	height = img1_rgb.shape[0]
	width = img1_rgb.shape[1]
	if page in frame:
		img_rgb = img1_rgb[frame[page][1]:height-frame[page][1], 0:width]
		img_result = img1_result[frame[page][1]:height-frame[page][1], 0:width]
	else:
		img_rgb = img1_rgb
		img_result = img1_result

	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	template = cv2.imread('mark-basmallah.png',0)

	w, h = template.shape[::-1]
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
	b = []
	r = {}
	for pt in zip(*loc[::-1]):
		sensitivity = 100
		sensitivityX = round(pt[0]/sensitivity)
		sensitivityY = round(pt[1]/sensitivity)


		if (sensitivityX, sensitivityY) not in f:
			if sensitivityY not in r:
				r[int(sensitivityY)] = []

			cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

			r[int(sensitivityY)].append(pt)
			result.append(pt)
			f.add((sensitivityX, sensitivityY))

	cv2.imwrite('header.png'.format(i),img_rgb)
	print('found header = {}'.format(len(f)))




def matchTemplate(imgName,page):
	img_rgb = cv2.imread(imgName)
	img_result = cv2.imread(imgName)
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	template = cv2.imread('mark.png',0)

	marginX = 0
	marginY = 0
	marginAdd = 0
	if page in frame:
		marginX = frame[page][0]
		marginY = frame[page][1]
		marginAdd = frame[page][2]

	print(marginX)

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
	b = []
	r = {}
	for pt in zip(*loc[::-1]):
		sensitivity = 100
		sensitivityX = round(pt[0]/sensitivity)
		sensitivityY = round(pt[1]/sensitivity)


		if (sensitivityX, sensitivityY) not in f:
			if sensitivityY not in r:
				r[int(sensitivityY)] = []

			r[int(sensitivityY)].append(pt)
			f.add((sensitivityX, sensitivityY))
			#cv2.circle(img_rgb, pt, 3, (0, 0, 255), -1)
			#cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

	for listpt in r.values():
		listpt.sort(key=sort_x,reverse = True)
		for pt in listpt:
			b.append(pt)

	for pt in b:
		i+=1
		imgCopy = img_result.copy()

		nodes = []
		#p1 = (pt[0]-5, pt[1]-5)
		if i==1:
			prevpt = (width - 5 - marginX - marginAdd,ymin - 20 + marginY)
		#	if pt[1]-5 + marginY>ymin:
		#		p1 = (xmin - 35, ymin - 25 + marginY)
		#	p2 = (p1[0] + marginX, pt[1]-5)
		#	p3 = (pt[0]-5, pt[1]-5)
		#	p4 = (pt[0]-5, pt[1] + h + 5)
		#	p5 = (width - 5 - marginX, pt[1] + h + 5)
		#	p6 = (width - 5 - marginX, p1[1])
		#	nodes.append(p1)
		#	nodes.append(p2)
		#	nodes.append(p3)
		#	nodes.append(p4)
		#	nodes.append(p5)
		#	nodes.append(p6)
		#else:

		p1 = (prevpt[0]-5, prevpt[1]-5)
		if xmin + marginX + marginAdd > prevpt[0] - 5:
			p1 = (width - 10 - marginX, prevpt[1] + h + 10)
		if pt[1]-5 < p1[1] + 50:
			p2 = (pt[0]-5,p1[1])
			p3 = (pt[0]-5, pt[1] + h + 10)
			p4 = (p1[0], pt[1] + h + 10)
		else:
			p2 = (xmin - 35 + marginX,p1[1])
			p3 = (p2[0], pt[1] - 5)
			p4 = (pt[0] - 5, pt[1] - 5)
		
		if pt[1]-5 > p1[1] + 50:
			p5 = (pt[0] - 5, pt[1] + h + 5)
			p6 = (width - 10 - marginX, pt[1] + h + 5)
			p7 = (width - 10 - marginX, p1[1] + h + 10)
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

	
	#cv2.imshow("Game Boy Screen", image)
	#cv2.waitKey(0)
	cv2.imwrite('res1.png',image)
	return edged;


filename = "DQ_{}.png".format(sys.argv[1].rjust(3, '0'))
print(filename)
matchTemplate(filename,int(sys.argv[1]))
matchTemplateHeader(filename,int(sys.argv[1]))
print(getListAyahFromPage(1))
#createTable()
#findContour(sys.argv[1])

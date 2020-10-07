import numpy as np
import cv2
from opts import get_opts
from matchPics import matchPics
from loadVid import loadVid
from planarH import compositeH, computeH_ransac

#Write script for Q3.1

opts = get_opts()

#load vids and template image
panda_source = loadVid("../data/ar_source.mov")
book_source = loadVid("../data/book.mov")
cv_cover = cv2.imread('../data/cv_cover.jpg')


for frame in range (book_source.shape[0]):
    # assigned each video frame to variable: book_frame
    print(frame) #print frame number to track processing
    book_frame = book_source[frame]

    #compute matches
    matches, locs1, locs2 = matchPics(book_frame, cv_cover, opts)

    #compute ransac
    matched_locs1 = np.array([locs1[i] for i in matches[:0]])
    matched_locs2 = np.array([locs2[i] for i in matches[:1]]) 

    bestH2to1, inliers = computeH_ransac(matched_locs1, matched_locs2, opts) # buggy computeH.ranac function, unable to produce output
    #bestH2to1, inlirs = cv2.findHomography(matched_locs1, matched_locs2, cv2.RANSAC,5.0) - for debugging

    #crop and resize
    width_newPanda = int(cv.cover.shape[1]/cv_cover.shape[0]*panda_source.shape[1]) # w x h
    crop = int(panda_source.shape[2]- width_newPanda)//2
    ##crop = int(panda_source.shape[2]- width_newPanda)/2 for debugging
    resized_panda = cv2.resize(cropped,panda,(cv_cover.shape[1],cv_cover.shape[0]))

    #compose composite pic
    compsiteFrame = composite(bestH2to1, resized_panda, book_frame)

    #write composite pic into output video
    fourcc=cv2.VideoWriter_fourcc(*'MP4V')
    writeVid = cv2.VideoWriter('ar.avi', fourcc, 25, (book_source.shape[1],book_source.shape[0]))
    writeVid.write(compositeFrame)

writeVid.release()
cv2.destroyAllWindows()



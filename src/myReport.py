import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

# Read in cars and notcars
vehicle_images_location = glob.glob('../../data/vehicles/vehicles/*/*.png')
non_vehicle_images_location = glob.glob('../../data/non-vehicles/non-vehicles/*/*.png')
print("Original num: {} car data and {} notcar data ...".format(
        len(vehicle_images_location), len(non_vehicle_images_location)))

# Augment fake data
def augment_fake_data(Locations):
    data=[]
    for loc in Locations:
        img=mpimg.imread(loc)
        fakeImg=cv2.flip(img,1)
        data.append(img)
        data.append(fakeImg)
    return data

cars=augment_fake_data(vehicle_images_location)
notcars=augment_fake_data(non_vehicle_images_location)
print("Augmented num: {} car data and {} notcar data ...".format(
        len(cars), len(notcars)))

# average the positive and negative data
#sample_size=min(len(cars), len(notcars))
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]

def randImg_car_notcar(cars,notcars):
    rand_num=np.random.randint(0, 300)
    car_img=cars[rand_num]
    notcar_img=notcars[rand_num]
    return car_img, notcar_img

car_img, notcar_img = randImg_car_notcar(cars,notcars)

from lesson_functions import *
img_titles=['Car','Not-Car']
mulImg_show_plt(1,2,img_titles,True, car_img,notcar_img)


# Color space features and HOG feature
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

# set HOG parameters
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off

# YCrCb color space 
colsp_car_img=convert_color(car_img,color_space)
colsp_notcar_img=convert_color(notcar_img,color_space)

def chancel_sperate(oriImg):
    num_chan=oriImg.shape[2]
    chan_imgs=[]
    for i in range(num_chan):
        chan_imgs.append(oriImg[:,:,i])
    return chan_imgs

chan_car_imgs=chancel_sperate(colsp_car_img)
chan_notcar_imgs=chancel_sperate(colsp_notcar_img)

# All color channels' HOG figure
print("Now choose color space: ",color_space)
for dep_i in range(len(chan_car_imgs)): 
    _,hog_im_car=get_hog_features(chan_car_imgs[dep_i], orient, pix_per_cell,
                                  cell_per_block,vis=True,feature_vec=True)
    _,hog_im_notcar=get_hog_features(chan_notcar_imgs[dep_i], orient,
                                     pix_per_cell,cell_per_block,vis=True,
                                     feature_vec=True)
#    car_feature=cv2.resize(chan_car_imgs[dep_i],(32,32))
#    notcar_feature=cv2.resize(chan_notcar_imgs[dep_i],(32,32))
    img_titles=['Car_CH-{}'.format(dep_i+1),'Car_CH-{}_HOG'.format(dep_i+1),
                'Not-Car_CH-{}'.format(dep_i+1),'Not-Car_CH-{}_HOG'.format(dep_i+1)]
    mulImg_show_plt(1,4,img_titles,True, chan_car_imgs[dep_i],hog_im_car,
                chan_notcar_imgs[dep_i],hog_im_notcar)

#%%
# trained a SVM classifier

from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

X_scaler = StandardScaler().fit(X_train)
if spatial_feat or hist_feat:
    # Fit a per-column scaler
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)


print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print("Color space is ", color_space)
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

#%%
# Check the prediction time for a single sample
t=time.time()
image = mpimg.imread('../test_images/test1.jpg')
#image = mpimg.imread('../test_images/test3.jpg')
draw_image = np.copy(image)
dectArea=np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
if np.max(image)>250 :
    image = image.astype(np.float32)/255

def draw_area(img, y_range,scale):
    color=(np.random.rand(1,3)*255).astype(np.int32).flatten().tolist()
    color2=tuple(color)
    print("{}-Line color is {}".format(scale-1, color2))
    cv2.rectangle(img, (0,y_range[0]), (img.shape[1],y_range[1]), color2, 6)
    return color2

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features=np.array(features).reshape(1, -1)
        if spatial_feat or hist_feat:
            test_features = scaler.transform(test_features)
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

scale_num=5
y_start=400
def set_search_areas(y_start=y_start, scale_num=scale_num):
    y_range=[]
    windows=[]
    for scale in range(1,scale_num+1):
        win_size=64*scale
        print("Window size is ",win_size, end=', ')
        y_start_stop_n = [y_start, y_start+ win_size] # Min and max in y to search in slide_window()
        line_color=draw_area(dectArea,y_start_stop_n,scale)
        y_range.append((y_start, y_start+ win_size,line_color))
        scale_wins=slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_n, 
                        xy_window=(win_size, win_size), xy_overlap=(0.5, 0.5),color=line_color)
        windows.extend(scale_wins)
    return y_range,windows

y_range,windows=set_search_areas(y_start,scale_num)

t=time.time()
hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

window_img = draw_boxes(draw_image, hot_windows, thick=6)                    
t2 = time.time()
print(round(t2-t, 2), 'Seconds to draw rectangle...')

img_titles=['Search_Area','{}_Detection_result'.format(color_space)]
mulImg_show_plt(1,2,img_titles,True, dectArea,window_img)


#%%
# Define a single function that can extract features using hog sub-sampling and make predictions
def search_windows_HOG_once(img, y_ranges,clf, X_scaler,color_space, orient, hog_channel,
              pix_per_cell, cell_per_block, spatial_size, hist_bins,spatial_feat=spatial_feat,
              hist_feat=hist_feat, hog_feat=hog_feat):
    boxes = []
    detected_boxes =[]
    if np.max(img)>2 :
        img = img.astype(np.float32)/np.max(img)
    for y_range in y_ranges:
        ystart, ystop, color = y_range
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = convert_color(img_tosearch, color_space)
        scale=(ystop-ystart+1)/64
#        scale=1.5
        if scale != 1.0 :
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                         (int(np.ceil(imshape[1]/scale)),
                                                      int(np.ceil(imshape[0]/scale))))
#            print("ctrans_tosearch size : ", ctrans_tosearch.shape)
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
        
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = orient*cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
        
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        if hog_channel == 'ALL':
            hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
#        print("nxsteps : {}\nnysteps: {}".format(nxsteps, nysteps))
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                if hog_channel == 'ALL':
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_features = hog_feat1
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
    
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                img_features=[]
                # Get color features
                if spatial_feat == True:
                    spatial_features = bin_spatial(subimg, size=spatial_size)
                    img_features.append(spatial_features)
                if hist_feat == True:
                    hist_features = color_hist(subimg, nbins=hist_bins)
                    img_features.append(hist_features)
                img_features.append(hog_features)
                # Scale features and make a prediction
                test_features = np.concatenate(img_features).reshape(1, -1)
                if spatial_feat or hist_feat:
                    test_features = X_scaler.transform(test_features)    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = clf.predict(test_features)
                
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(( (xbox_left, ytop_draw+ystart),
                            (xbox_left+win_draw,ytop_draw+win_draw+ystart),color )) 
                
                if test_prediction == 1:
                    detected_boxes.append(( (xbox_left, ytop_draw+ystart),
                            (xbox_left+win_draw,ytop_draw+win_draw+ystart),color )) 
                
    return detected_boxes,boxes

t=time.time()
hot_windows, windows=search_windows_HOG_once(image,y_range, svc, X_scaler, color_space,orient,hog_channel,pix_per_cell, cell_per_block,spatial_size, hist_bins)
window_img = draw_boxes(draw_image, windows, thick=6)  
window_img2 = draw_boxes(draw_image, hot_windows, thick=6)  
t2 = time.time()
print(round(t2-t, 2), 'Seconds to draw rectangle...')

img_titles=['Search_Area','{}_Detection_result2'.format(color_space)]
mulImg_show_plt(1,2,img_titles,True, window_img,window_img2)

#%%
# add heat map to identify vehicle positions
from scipy.ndimage.measurements import label
def get_heat_map(image, cur_boxes, pre_result_box=None):
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    for box in cur_boxes:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    threshold= max(np.max(heatmap)//2,1)
    if not pre_result_box is None:
        for box in pre_result_box:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
    heatmap[heatmap <= threshold] = 0
    heatmap = np.clip(heatmap, 0, 255)
    labels = label(heatmap)
#    print("threshold is {}, image maxum value is {}".format(threshold, np.max(image)) )
    
    # draw final results
    boxes=[]
    imcopy = np.copy(image)
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        boxes.append(bbox)
        # Draw the box on the image
        color=(1,0,1)
        if np.max(image)>2:
            color=(255,0,255)
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, 6)
    return imcopy, heatmap, boxes
result_img, heatmap,_ = get_heat_map(image,hot_windows)
img_titles=['Before_selected', 'Heatmap', 'Result']
mulImg_show_plt(1,3,img_titles,True, window_img2, heatmap, result_img)

# Test on the test_images
from os.path import split
def find_cars(img,pre_box=None, y_ranges=y_range,clf=svc, X_scaler=X_scaler,color_space=color_space, 
              orient=orient, hog_channel=hog_channel,pix_per_cell=pix_per_cell, 
              cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins):
    draw_image = np.copy(img)
    t=time.time()
    hot_windows, windows=search_windows_HOG_once(img,y_range, svc, X_scaler, color_space,orient,hog_channel,
                          pix_per_cell, cell_per_block,spatial_size, hist_bins)
    window_img = draw_boxes(draw_image, hot_windows, thick=6) 
    result_img, heatmap, result_box = get_heat_map(img,hot_windows, pre_box)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to detect Vehicles...')
    
    return window_img, heatmap, result_img,result_box

test_image_loc=glob.glob('../test_images/*.jpg')
for testLoc in test_image_loc:
    _, fileName=split(testLoc)
    img=mpimg.imread(testLoc)
    ori_img, heat_map, result_img, _=find_cars(img)
    img_titles=['{}'.format(fileName), 'Detect_area','Heatmap', 'Result_{}'.format(fileName)]
    mulImg_show_plt(1,4,img_titles,True, img,ori_img, heat_map, result_img)

#%%
from moviepy.editor import VideoFileClip
import os
# Test on videos
def process_video(v_loc,func, save_type='.mp4'):
    video0 = cv2.VideoCapture(v_loc)
    cols=2
    
    ret, frame = video0.read()
    if not ret:
        print("Can't open video")
        return
#    frame= cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
    ori_img, heat_map, result_img,pre_boxes =func(frame)
    outsave= mulImg_show_cv(1,cols,img_titles,False,
                                 ori_img ,result_img)
    size = (outsave.shape[1],outsave.shape[0])
    fps =video0.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('D','I','B',' ')
    out = cv2.VideoWriter('../output.avi',fourcc, fps, size,True)
    
    while(video0.isOpened()): 
        ret, frame = video0.read()
        if not ret: break
#        frame= cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
        ori_img, heat_map, result_img, pre_boxes=func(frame,pre_boxes)
        outsave= mulImg_show_cv(1,cols,img_titles,False,
                                 ori_img ,result_img) 
        out.write(outsave)
        k = cv2.waitKey(3)
        # ESC
        if (k & 0xff == 27):  
            break
    else:
        print("Can't open video")
    video0.release()
    out.release()
    cv2.destroyAllWindows()
    
    # transform movie from *.avi to *.mp4
    _, fileName=split(v_loc)
    myclip = VideoFileClip("../output.avi")
    print ("fps is {}, duration is {}, end is {}".format(myclip.fps, 
           myclip.size, myclip.end) )
    myclip.write_videofile("../output_{}".format(fileName), fps=myclip.fps) # export as video
    myclip.close
    
    # remove *.avi for saving storage
    oldFile="../output.avi"
    if os.path.exists(oldFile): 
        os.remove(oldFile)
        print("Now removed %s"%oldFile)
    else:
        print('no such file:%s'%oldFile)
    return 

#process_video("../test_video.mp4",find_cars)
process_video("../project_video.mp4",find_cars)

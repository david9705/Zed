#调用ZED自带的库
#import pyzed.camera as zcam
#import pyzed.defines as sl
#import pyzed.types as tp
#import pyzed.core as core
import math
import numpy as np
import sys
import cv2
import pyzed.sl as sl

# Surf detect
#设置SURF的参数
MIN_MATCH_COUNT = 30
surf = cv2.xfeatures2d.SURF_create()
FLANN_INDEX_KDITREE = 0
flannParam = dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})
#读取模板图像
trainImg=cv2.imread("0.jpg",0)
trainKP,trainDesc=surf.detectAndCompute(trainImg,None)

font = cv2.FONT_HERSHEY_SIMPLEX
 


def main():
    #ZED自带程序，照搬过来
    print("Running...")
    #init = zcam.PyInitParameters()
    #zed = zcam.PyZEDCamera()

    '''new'''
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD2K

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    '''new'''
    

    '''
    #获取深度，点云等数据
    runtime = zcam.PyRuntimeParameters()
    mat = core.PyMat()
    depth = core.PyMat()
    point_cloud = core.PyMat()
    '''

    '''new'''
    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
    # Setting the depth confidence parameters
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.textureness_confidence_threshold = 100

    # Capture 150 images and depth, then stop
    i = 0
    mat = sl.Mat()  #image -> mat
    depth = sl.Mat()
    point_cloud = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m
    '''new'''

    '''
    #ZED的参数设置
    init_params = zcam.PyInitParameters()
    init_params.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.PyUNIT.PyUNIT_MILLIMETER  # Use milliliter units (for depth measurements)
    #改变相机的模式，VGA分辨率低但是速度会快很多
    init_params.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_VGA 
    init_params.camera_fps = 100
    '''

    
    key = ''
    while key != 113:  # for 'q' key
        #err = zed.grab(runtime)
        #if err == tp.PyERROR_CODE.PySUCCESS:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(mat, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            
            frame = mat.get_data()
            t1 = cv2.getTickCount()
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #因为ZED是双目相机，所以这里识别部分只使用左镜头的图像
            frame = cv2.resize(frame, (int(mat.get_width()/2),int(mat.get_height()/2)),interpolation=cv2.INTER_AREA)
            # SURF detect 
            KP,Desc = surf.detectAndCompute(frame,None)
            matches = flann.knnMatch(Desc,trainDesc,k=2)

            goodMatch=[]
            for m,n in matches:
                if(m.distance<0.8*n.distance):
                    goodMatch.append(m)
            if(len(goodMatch)>MIN_MATCH_COUNT):
                yp=[]
                qp=[]
                for m in goodMatch:
                    yp.append(trainKP[m.trainIdx].pt)
                    qp.append(KP[m.queryIdx].pt)
                yp,qp=np.float32((yp,qp))
                H,status=cv2.findHomography(yp,qp,cv2.RANSAC,3.0)
                h,w=trainImg.shape
                trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
                imgBorder=cv2.perspectiveTransform(trainBorder,H)
                cv2.polylines(frame,[np.int32(imgBorder)],True, (0,255,0),3)#在imshow的图像上显示出识别的框
                # get coordinate 获得目标物体bbox的坐标并计算出中心点坐标
                c1 = imgBorder[0,0]
                c2 = imgBorder[0,1]
                c3 = imgBorder[0,2]
                c4 = imgBorder[0,3]
                xmin = min(c1[0],c2[0])
                xmax = max(c3[0],c4[0])
                ymin = min(c1[1],c4[1])
                ymax = max(c2[1],c3[1])
                
                #distance_point_cloud
                x = round(xmin+xmax)
                y = round(ymin+ymax)

                if(x < 0 or y < 0):
                    x = 0
                    y = 0
                
                #计算出的中心点坐标后，来获取点云数据
                err, point_cloud_value = point_cloud.get_value(x, y)
                #由点云数据计算出和相机的距离（左相机为原点）
                distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])
                #把距离打在屏幕里


                if not np.isnan(distance) and not np.isinf(distance):
                    distance = round(distance)
                    print("Distance to Camera at ({0}, {1}): {2} mm\n".format(x, y, distance))
                    Z = "distance:{} mm".format(distance)
                    cv2.putText(frame,Z,(xmax,ymax), font, 0.7,(255,255,255),2,cv2.LINE_AA)

                # Increment the loop
                else:
                    print("Can't estimate distance at this position, move the camera\n")
            else:
                print("Not Eough match") 
            sys.stdout.flush()
            t2 = cv2.getTickCount()
            fps = cv2.getTickFrequency()/(t2-t1)
            fps = "Camera FPS: {0}.".format(fps)
            cv2.putText(frame,fps,(25,25),font,0.5,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow("ZED", frame)
            key = cv2.waitKey(5)
        else:
            key = cv2.waitKey(5)
    
    cv2.destroyAllWindows()

    zed.close()
    print("\nFINISH")

if __name__ == "__main__":
    main()

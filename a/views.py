from json import dumps
from django.shortcuts import render
# from sympy import primenu, principal_branch
from tensorflow import keras
from .models import *
from .serializers import ModelsSerializer, SampleSerializer
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import math
import time
from a.m import *
from example.utils import *
import glob
import os
import matlab.engine
from PIL import Image
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet
from rest_framework import status
from keras.preprocessing import image
from rest_framework.decorators import api_view
import cv2
from . import Load_Model


simp_var= Load_Model.l_model()
new_dic=dict()
new_dic_mob=dict()
def get_occ_imgs(img, img_size, occ_size, occ_pixel, occ_stride, classes, model):
    # Get original image
   
    input_image = Image.open(img)
    input_image = input_image.resize((224, 224))
    input_image = np.asarray(input_image)
    input_image = input_image.astype('float32')
    # input_image = input_image / 255  # normalized to [0,1]

    image = input_image
    # Index of class with highest probability
    class_index = np.argmax(classes)
    print('True class index:', class_index)

    # Define number of occlusions in both dimensions
    output_height = int(math.ceil((img_size - occ_size) / occ_stride + 1))
    output_width = int(math.ceil((img_size - occ_size) / occ_stride + 1))
    print('Total iterations:', output_height, '*', output_width, '=', output_height * output_width)

    # Initialize probability heatmap and occluded images
    temp_img_list = []
    prob_matrix = np.zeros((output_height, output_width))

    start = time.time()

    for h in range(output_height):
        for w in range(output_width):
            # Occluder window:
            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(img_size, h_start + occ_size)
            w_end = min(img_size, w_start + occ_size)

            # Getting the image copy, applying the occluding window and classifying it:
            occ_image = image.copy()
            occ_image[h_start:h_end, w_start:w_end, :] = occ_pixel
            occ_image = occ_image.reshape(1, 224, 224, 3)
            predictions = pred_prob_list(model, occ_image.copy())[0]
            prob = predictions[class_index]

            # Collect the probability value in a matrix
            prob_matrix[h, w] = prob

            # Collect occluded images
            # occ_image[h_start:h_end, w_start:w_end, :] = prob*255
            # cv2.putText(img=occ_image, text=str(prob), org=(w_start, int(h_start + (h_end - h_start) / 2)),
            #           fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2, color=(255*(1-prob),255*(1-prob),255*(1-prob)), thickness=1
            #           )
            # cv2.imwrite('occ_exp/video/'+'person'+str(h*output_width+w+1).zfill(6)+'.png',occ_image)

            # To save occluded images as a video, run the following shell command
            """ffmpeg -framerate 5 -i occ_exp/video/<img_name>%06d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p occ_exp/<img_name>.mp4"""

            # temp_img_list.append(occ_image)

        print('Percentage done :', round(((h + 1) * output_width) * 100 / (output_height * output_width), 2), '%')

    end = time.time()
    elapsed = end - start
    print('Total time taken:', elapsed, 'sec\tAverage:', elapsed / (output_height * output_width), 'sec')

    # Save probabilities and all occluded images in one
    # np.save('occ_exp/probs_' + img_name + '.npy', prob_matrix)
    # save_occs(temp_img_list, img_size, img_size, img_path.split('/')[-1])

    return prob_matrix

def regularize(prob, norm, percentile):
    # First save the original prob matrix as heat-map
    # f = plt.figure(1)
    # sns.heatmap(prob, xticklabels=False, yticklabels=False)
    # f.show()
    # f.savefig('occ_exp/heatmap_' + 'person')

    # Apply Regularization
    prob = normalize_clip(prob) if norm else prob
    clipped = clip_weak_pixel_regularization(prob, percentile=percentile)
    reg_heat = blur_regularization(1 - clipped, size=(3, 3))
    # Save regularized heat-map
    # f2 = plt.figure(2)
    # sns.heatmap(reg_heat, xticklabels=False, yticklabels=False)
    # f2.savefig('occ_exp/heatmap_reg_' + 'person')

    return reg_heat

def join(heat_reg, img, img_size, occ_size, htmPath):
    # Get original image
    # cv2.imshow('image', img)
    image = cv2.imread(img, 1)
    # print('Original Dimensions : ', image.shape )
    inp_img = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

    H, W = image.shape[0], image.shape[1]
    # print(H)
    # print(W)

    bord = int(occ_size / 2)

    # Define heat-map to be projected on original image
    heat_map = cv2.resize(heat_reg, (img_size, img_size)).astype(np.float32)

    # Second way to define heat-map - manually set border values
    # heat_map = np.zeros((img_size, img_size))
    # heat_map[bord:img_size - bord, bord:img_size - bord] = cv2.resize(heat_reg,
    #     (img_size - occ_size, img_size - occ_size)).astype(np.float32)
    # np.place(heat_map, heat_map == 0.0, np.median(heat_map))

    # Third way to define heat-map - replicate border values
    # heatmap = cv2.resize(heat, (img_size-occ_size, img_size-occ_size)).astype(np.float32)
    # heatmap = cv2.copyMakeBorder(heat-map,bord,bord,bord,bord,cv2.BORDER_REPLICATE)

    # Original image * heat-map
    for i in range(3):
        inp_img[:, :, i] = heat_map * inp_img[:, :, i]
    inp_viz = cv2.resize(inp_img, (W, H))

    print(htmPath,"this url ")
    print(img.split('/')[-1])
    cv2.imwrite(htmPath + img.split('/')[-1], inp_viz)
    print("image writed done!!!")
    # plt.axis('off')
    # plt.imshow(np.uint8(inp_viz))
    # plt.imshow(np.uint8(inp_viz), cmap='jet', alpha=0.4)
    # cam = cv2.applyColorMap(inp_viz, cv2.COLORMAP_RAINBOW)
    # cv2.imwrite('occ_exp/heatmap_25_Rain/' + img.split('/')[-1], cam)
    # plt.savefig('occ_exp/' + img.split('/')[-1])

    return inp_viz

def my_normalize(cam):
    max_i = np.max(cam)
    min_i = np.min(cam)
    cam = (255 * (cam - min_i)) / (max_i - min_i)
    return cam

def load_image(path, preprocess=True):
    """Load and preprocess image."""
    # x = image.load_img(path, target_size=(H, W))
    original_x = image.load_img(path)
    img_size = original_x.size
    img_size = img_size[0]
    x = image.load_img(path, target_size=(224, 224))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        # x = x * 1. /255
    return x, original_x, img_size

# Create your views here.
def listOfSample():
    sample = Sample.objects.all()
    ser = SampleSerializer(sample, many=True)

    return ser.data

def listofModel():
    model = Models.objects.all()  # return models
    ser = ModelsSerializer(model, many=True)
    return ser.data

def renderPage(request):
    return render(request, 'a/index.html')

def index(request):
    # list of Mosel usage
    model = listofModel()
    # print(model)
    sampels = listOfSample()
    # print(sampels)

    # list of dictionary contain model&sample&predict
    content = {
        'status': True,
        'response': {
            "dis": model,
            "rows_list": sampels
        }

    }
    return JsonResponse(content)


@csrf_exempt
def firstStep_1(request):
   global simp_var
   global  new_dic

   data = request.POST
   model = data['model']
   # print(data)
   pic = request.FILES['image']
   hh = request.FILES['image'].name

   smodel = Models.objects.get(name=model)
   s = Sample(orgImg=pic, withModel=smodel)
   s.save()
   sId = s.id
   sUrl = s.orgImg
   print(sId)
   print(sUrl)

   # TODO orginal image
   original_img_path = 'media/orginal/' + hh
   preprosses_img_path = 'media/preprosses/' + hh

   print(original_img_path)
   print(preprosses_img_path)
   htmPath = 'media/heatmap/'

  
   # try:
   if simp_var is None:

        simp_var=Load_Model.l_model()
        print(simp_var.model)
   else:
      
        print("it work")


  

    #  preprossess image--
   
   eng = matlab.engine.start_matlab()
   eng.cd(r'C:\Users\Fatima', nargout=0)
   eng.test(original_img_path, nargout=0)
   
   
   if os.path.isfile(htmPath) == 0:
        img_size = 224
        print('hereeeeeeeeeeeeee')
        input_image = None
        input_image = Image.open(preprosses_img_path)
        print(input_image)
        print('done read image')
        input_image = input_image.resize((224, 224))
        print('resize ok.')
        input_image = np.asarray(input_image)
        input_image = input_image.astype('float32')
        input_image = input_image.reshape(1, 224, 224, 3)
        print('1 step finish')
        # Get probability list and print top 5 classes
        mm=simp_var.model
        result = pred_prob_list(mm, input_image)
        new_dic[sId]=result
        print('thisss')
        print(new_dic[sId])
        print("end thisss")
        class_index = np.argmax(result)
        print(class_index)

        if class_index:

            s.predict = 'True'
            s.save()
        else:

            s.predict = 'False'
            s.save()

        print('after resulat')

        a = 0
        if a == 0:
            # get object
            sample = Sample.objects.get(pk=sId)
            ser = SampleSerializer(sample)

            content = {
                'status': True,
                'response': {

                    "info": ser.data
                }

            }
            return JsonResponse(content)



@csrf_exempt
def firstStep_2(request):
   global simp_var
   global new_dic

   print(new_dic)
   data = request.POST
   predict = data['predict']
   pic = request.FILES['image']
   sId =int(data['id'])

   print(new_dic[sId])
   hh = request.FILES['image'].name

   sam = Sample.objects.get(pk=sId)
   sam.predict = predict
   sam.save()

   original_img_path = 'media/orginal/' + hh
   preprosses_img_path = 'media/preprosses/' + hh
   htmPath = 'media/heatmap/'

   print(original_img_path)
   print(preprosses_img_path)

   # try:
   if simp_var is None:
        print("dont work")
        simp_var = Load_Model.l_model()
        print(simp_var)
   else:
        print("it worked")

   occ_size, occ_pixel, occ_stride = 100, 0, 100

   # object = Load_Model.l_model()
   # model = object.model

   if os.path.isfile(htmPath) == 0:
        img_size = 224
        print('hereeeeeeeeeeeeeeeeeeeeeeeeee')

        result=new_dic[sId]
        print(result)


        a = 0
        if a == 0:
            probs = get_occ_imgs(preprosses_img_path, img_size, occ_size, occ_pixel, occ_stride, result,simp_var.model)
            print('probs ok')
            # print(probs)

            heat = regularize(probs, 1, 10)
            print('heat ok')
            # Project heatmap on original image

            aug = join(heat, original_img_path, img_size, occ_size, htmPath)
            # print(aug)
            print('\nDone')
            # TODO
            pic = ('/media/heatmap/' + hh)
            print(pic)
            sam = Sample.objects.get(pk=sId)
            sam.heatMap = pic
            sam.save()
            print(sam.id)
            print(sam.heatMap,"saved")

            # get object
            ser = SampleSerializer(sam)
            new_dic.clear()
            content = {
                'status': True,
                'response': {
                    # "dis": model,
                    # "rows_list": sampels,
                    "info": ser.data
                }

            }
            return JsonResponse(content)


@csrf_exempt
def firstStep_3(request):
    data = request.POST
    predict = data['predict']
    id = data['id']

    sam = Sample.objects.get(pk=id)
    sam.predict = predict
    sam.save()
    new_dic.clear()
    content = {
        'status': True,
        'response': []

    }
    return JsonResponse(content)

# mobile api
class DBViewSet(ModelViewSet):

    @api_view(["POST"])
    def submit1(request):
        global simp_var
        global new_dic_mob

        data = request.data
        model = data['model']
        print(model)
        pic = request.FILES['image']
        hh = request.FILES['image'].name
        # print(pic)
        # print(pic)
        smodel = Models.objects.get(name=model)
        s = Sample(orgImg=pic, withModel=smodel)
        s.save()
        sId = s.id
        sUrl = s.orgImg
        # print(sId)
        # print(sUrl)

        original_img_path = 'media/orginal/' + hh
        preprosses_img_path = 'media/preprosses/' + hh
        print(original_img_path)


        if simp_var is None:

            simp_var = Load_Model.l_model()
            print(simp_var.model)
        else:
            # simp_var.model._make_predict_function()
            print("it work")


        print(preprosses_img_path)


        #  preprossess image--
        #   TODO
        eng = matlab.engine.start_matlab()
        eng.cd(r'C:\Users\Fatima', nargout=0)
        eng.test(original_img_path, nargout=0)
        print('script ok')


        htmPath = 'media/heatmap/'
        if os.path.isfile(htmPath) == 0:
            img_size = 224
            print('hereeeeeeeeeeeeeeeeeeeeeeeeee')
            input_image = None
            input_image = Image.open(preprosses_img_path)
            input_image = input_image.resize((224, 224))
            print('resize ok.')
            input_image = np.asarray(input_image)
            input_image = input_image.astype('float32')
            # input_image = input_image / 255  # normalized to [0,1]
            input_image = input_image.reshape(1, 224, 224, 3)
            print('1 step finish')
            # Get probability list and print top 5 classes
            result = pred_prob_list(simp_var.model, input_image)
            new_dic_mob[sId] = result
            print(new_dic)
            class_index = np.argmax(result)
            print(class_index)

            if class_index:

                s.predict = 'True'
                s.save()
            else:

                s.predict = 'False'
                s.save()

            print('after resulat')

            a = 0
            if a == 0:
                # K.clear_session()
                # get object
                sample = Sample.objects.get(pk=sId)
                ser = SampleSerializer(sample)

                content = {
                    'status': True,
                    'response': {

                        "info": ser.data
                    }

                }
                return JsonResponse(content)

    @api_view(["POST"])
    def submit2(request):
        global new_dic_mob
        global simp_var

        data = request.data
        predict = data['predict']
        print(predict)
        sId = int(data['id'])
        print(sId)
        hh = data['image']
        print(hh)

        sam = Sample.objects.get(pk=sId)
        sam.predict = predict
        sam.save()

        original_img_path = 'media/orginal/' + hh
        preprosses_img_path = 'media/preprosses/' + hh
        print(original_img_path)
        print(preprosses_img_path)


        if simp_var is None:
            print("dont work")
            simp_var = Load_Model.l_model()
            print(simp_var)
        else:
            print("it worked")


        occ_size, occ_pixel, occ_stride = 100, 0, 100

        htmPath = 'media/heatmap/'
        if os.path.isfile(htmPath) == 0:
            img_size = 224
            result = new_dic_mob[sId]
            print(result)
            a = 0
            if a == 0:
                probs = get_occ_imgs(preprosses_img_path, img_size, occ_size, occ_pixel, occ_stride, result,simp_var.model )
                print('probs ok')
                # print(probs)

                heat = regularize(probs, 1, 10)
                print('heat ok')
                # Project heatmap on original image

                aug = join(heat, original_img_path, img_size, occ_size, htmPath)
                print('\nDone')
                # TODO
                pic = ('/media/heatmap/' + hh)
                sam = Sample.objects.get(pk=sId)
                sam.heatMap = pic
                sam.save()
                print(sam.id)

                # get object
                ser = SampleSerializer(sam)
                new_dic_mob.clear()
                content = {
                    'status': True,
                    'response': {
                        # "dis": model,
                        # "rows_list": sampels,
                        "info": ser.data
                    }

                }
                return JsonResponse(content, status=status.HTTP_200_OK)

    @api_view(["POST"])
    def stop_mobile(request):

        data = request.data
        predict = data['predict']
        print(predict)
        id = data['id']
        print(id)

        sam = Sample.objects.get(pk=id)
        sam.predict = predict
        sam.save()
        new_dic.clear()
        content = {
            'status': True,
            'response': []

        }
        return JsonResponse(content)

    @api_view(["GET"])
    def start(request):
        print('fffffffffffffffff')
        # list of Mosel usage
        model = Models.objects.all()  # return models
        ser1 = ModelsSerializer(model, many=True)
        model = ser1.data
        # print(model)
        sample = Sample.objects.all()
        ser2 = SampleSerializer(sample, many=True)
        sampels = ser2.data
        # print(sampels)

        # list of dictionary contain model&sample&predict
        content = {
            'status': True,
            'response': {
                "dis": model,
                "rows_list": sampels
            }

        }
        return JsonResponse(content, status=status.HTTP_200_OK)

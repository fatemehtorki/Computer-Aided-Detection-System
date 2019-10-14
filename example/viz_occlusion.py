import math
import time
from a.m import *
from example.utils import *
import glob
import os
from PIL import Image

def get_occ_imgs(img, img_size, occ_size, occ_pixel, occ_stride, classes):
    # Get original image
    #image = cv2.imread(img)
    #image = cv2.resize(image, (img_size, img_size)).astype(np.float32)
    #input_image = image
    #input_image = None
    input_image = Image.open(img)
    input_image = input_image.resize((224, 224))
    input_image = np.asarray(input_image)
    input_image = input_image.astype('float32')
    #input_image = input_image / 255  # normalized to [0,1]

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
            #occ_image[h_start:h_end, w_start:w_end, :] = prob*255
            #cv2.putText(img=occ_image, text=str(prob), org=(w_start, int(h_start + (h_end - h_start) / 2)),
             #           fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2, color=(255*(1-prob),255*(1-prob),255*(1-prob)), thickness=1
              #           )
            #cv2.imwrite('occ_exp/video/'+'person'+str(h*output_width+w+1).zfill(6)+'.png',occ_image)
            
            # To save occluded images as a video, run the following shell command
            """ffmpeg -framerate 5 -i occ_exp/video/<img_name>%06d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p occ_exp/<img_name>.mp4"""

            #temp_img_list.append(occ_image)

        print('Percentage done :', round(((h + 1) * output_width) * 100 / (output_height * output_width), 2), '%')

    end = time.time()
    elapsed = end - start
    print('Total time taken:', elapsed, 'sec\tAverage:', elapsed / (output_height * output_width), 'sec')

    # Save probabilities and all occluded images in one
    #np.save('occ_exp/probs_' + img_name + '.npy', prob_matrix)
    # save_occs(temp_img_list, img_size, img_size, img_path.split('/')[-1])

    return prob_matrix


def regularize(prob, norm, percentile):
    # First save the original prob matrix as heat-map
    #f = plt.figure(1)
    #sns.heatmap(prob, xticklabels=False, yticklabels=False)
    #f.show()
    #f.savefig('occ_exp/heatmap_' + 'person')

    # Apply Regularization
    prob = normalize_clip(prob) if norm else prob
    clipped = clip_weak_pixel_regularization(prob, percentile=percentile)
    reg_heat = blur_regularization(1-clipped, size=(3, 3))
    # Save regularized heat-map
    #f2 = plt.figure(2)
    #sns.heatmap(reg_heat, xticklabels=False, yticklabels=False)
    #f2.savefig('occ_exp/heatmap_reg_' + 'person')

    return reg_heat


def  join(heat_reg,img,img_size, occ_size, htmPath):
    # Get original image
    # cv2.imshow('image', img)
    image = cv2.imread(img,1)
    # print('Original Dimensions : ', image.shape )
    inp_img = cv2.resize(image,(img_size, img_size),interpolation = cv2.INTER_AREA)
    H, W = image.shape[0], image.shape[1]
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
    print(img.split('/')[-1])
    print(htmPath)
    cv2.imwrite( htmPath + img.split('/')[-1], inp_viz)


    #plt.axis('off')
    #plt.imshow(np.uint8(inp_viz))
    #plt.imshow(np.uint8(inp_viz), cmap='jet', alpha=0.4)
    #cam = cv2.applyColorMap(inp_viz, cv2.COLORMAP_RAINBOW)
    #cv2.imwrite('occ_exp/heatmap_25_Rain/' + img.split('/')[-1], cam)
    #plt.savefig('occ_exp/' + img.split('/')[-1])

    return inp_viz

def my_normalize(cam):
    max_i = np.max(cam)
    min_i = np.min(cam)
    cam = (255*(cam - min_i)) / (max_i - min_i)
    return cam


from keras.preprocessing import image

def load_image(path, preprocess=True):
    """Load and preprocess image."""
    #x = image.load_img(path, target_size=(H, W))
    original_x = image.load_img(path)
    img_size = original_x.size
    img_size = img_size[0]
    x = image.load_img(path, target_size=(224, 224))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        #x = x * 1. /255
    return x, original_x, img_size


if __name__ == '__main__':
    #args = get_args()
    #print('\n', args)

    #img_name = 'person34_virus_76'
    #img_path = '/home/atlas/PycharmProjects/SimpleNet/cropped_ChestXRay/test/1/' + img_name + '.jpeg'  # sys.argv[1]
    #test_path = '/home/atlas/PycharmProjects/SimpleNet/CLAHEHM_Normal_Pneumonia/test/1/'
    #original_img_path = '/home/atlas/Desktop/htm/images/pneumonia/'#'/home/atlas/PycharmProjects/SimpleNet/chest_xray/test/PNEUMONIA/'
    # TODO orginal image
    original_img_path ='media/orginal/'
    # original_img_path ='E:/workspacedjango/image and code/test_original/PNEUMONIA/'



    occ_size, occ_pixel, occ_stride = 100, 0, 100

    # Input pre-trained model, defined in m.py
    model = load_trained_model()
    # TODO preprossess image
    for file_add in glob.glob('E:/workspacedjango/image and code/test_preprocced/1/*.JPEG'):
        # Get original image

        #print('file add = ',file_add)
        htmPath = 'media/heatmap/'
        if os.path.isfile(htmPath) == 0:
            img_size = 224

            input_image=None
            input_image = Image.open(file_add)
            input_image = input_image.resize((224, 224))
            input_image = np.asarray(input_image)
            input_image = input_image.astype('float32')
            #input_image = input_image / 255  # normalized to [0,1]

            input_image = input_image.reshape(1, 224, 224, 3)

            # Get probability list and print top 5 classes
            result = pred_prob_list(model, input_image)
            a=0
            if a==0:


                # Start occlusion experiment and store predicted probabilities in a file
               # print('Running occlusion iterations (Class:', de_result[0][1], ') ...\n')
                probs = get_occ_imgs(file_add, img_size, occ_size, occ_pixel, occ_stride, result)

                # Get probabilities and apply regularization
                # print('\nGetting probability heat-map and regularizing...')
                #probs = np.load('occ_exp/probs_' + img_name + '.npy')
                heat = regularize(probs, 1, 10)

                # Project heatmap on original image

                # print('\nProject the heat-map to original image...')
                original_img_path=original_img_path + file_add.split('\\')[-1]
                # print(original_img_path)

                aug = join(heat, original_img_path, img_size, occ_size, htmPath)

                print('\nDone')


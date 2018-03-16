import numpy as np
import cv2
import os
from glob import glob
import anogan

def combination(list_name):
        list_imgs = []
        for name in list_name:
            image = cv2.imread(name, 1)
            list_imgs.append(image)
        np_imgs = np.array(list_imgs)
        combined_image = anogan.combine_images(np_imgs)
        return combined_image

def save_all_result():
    for idx in range(10):
        list_diff_name = sorted(glob('./anomaly_result/diff_%d*.png'%idx))
        list_pred_name = sorted(glob('./anomaly_result/pred_%d*.png'%idx))
        list_query_name = sorted(glob('./anomaly_result/qurey_%d*.png'%idx))

        combined_diff = combination(list_diff_name)
        combined_pred = combination(list_pred_name)
        combined_query = combination(list_query_name)

        cv2.imwrite('./combined_result/result_diff_%d.png'%idx, combined_diff)
        combined_pred = cv2.resize(combined_pred, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        combined_query = cv2.resize(combined_query, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('./combined_result/result_pred_%d.png'%idx, combined_pred)
        cv2.imwrite('./combined_result/result_query_%d.png'%idx, combined_query)

def absolute_diff():
    for idx in range(10):
        list_pred_name = sorted(glob('./anomaly_result/pred_%d*.png'%idx))
        list_query_name = sorted(glob('./anomaly_result/qurey_%d*.png'%idx))

        combined_pred = combination(list_pred_name)
        combined_query = combination(list_query_name)

        combined_query_color = combined_query.copy()
        combined_query = cv2.cvtColor(combined_query, cv2.COLOR_BGR2GRAY)
        combined_pred = cv2.cvtColor(combined_pred, cv2.COLOR_BGR2GRAY)

        combined_query = combined_query.astype(np.float32)
        combined_pred = combined_pred.astype(np.float32)

        ad = (((combined_query - combined_pred)+255)/2).astype(np.uint8)
        # ad = np.where(ad <= int(255*0.65), 0, ad).astype(np.uint8)
        # cv2.imshow('debug', ad)
        # key = cv2.waitKey()
        # if key == 27:
        #     break
        # print (ad.max(), ad.min())
        ad_color = cv2.applyColorMap(ad, cv2.COLORMAP_JET)
        ad_color[:,:,0] = np.where(ad <= int(255*.65), np.where(ad >= int(255*.35), 0, ad_color[:,:,0]), ad_color[:,:,0])
        ad_color[:,:,1] = np.where(ad <= int(255*.65), np.where(ad >= int(255*.35), 0, ad_color[:,:,1]), ad_color[:,:,1])
        ad_color[:,:,2] = np.where(ad <= int(255*.65), np.where(ad >= int(255*.35), 0, ad_color[:,:,2]), ad_color[:,:,2])

        # ad_color[:,:,0] = np.where(ad >= int(255*.35), 0, ad_color[:,:,0])
        # ad_color[:,:,1] = np.where(ad >= int(255*.35), 0, ad_color[:,:,1])
        # ad_color[:,:,2] = np.where(ad >= int(255*.35), 0, ad_color[:,:,2])
        # print (type(ad>=int(255*0.45)))
        # print (type((ad>=int(255*0.45))[0] | (ad<=int(255*0.65))[0]))
        
        # print ((ad>=int(255*0.45))[0] & (ad<=int(255*0.65))[0])
        # print (ad>=int(255*0.55))
        # exit()

        # ad_color[:,:,0][(ad>=int(255*0.55))[0] & (ad>=int(255*0.45))[0]] = 0
        # ad_color[:,:,1][(ad>=int(255*0.55))[0] & (ad>=int(255*0.45))[0]] = 0
        # ad_color[:,:,2][(ad>=int(255*0.55))[0] & (ad>=int(255*0.45))[0]] = 0
        
        
        show = cv2.addWeighted(combined_query_color, 0.4, ad_color, 0.8, 0)
        show = cv2.resize(show, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

        # cv2.imwrite('anomaly_result/result_diff3_%d.png'%idx,show)


if __name__ == '__main__':
    save_all_result()
    # absolute_diff()
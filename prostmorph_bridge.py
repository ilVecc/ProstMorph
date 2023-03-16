import redis
import numpy as np
import cv2
import logging
import SimpleITK as sitk

from notebooks.setup import model, latest_checkpoint
from utils import ImageLoader


# model.load_weights(latest_checkpoint)


def argmin(size):
    arg = 0
    if size[1] < size[arg]:
        arg = 1
    if size[2] < size[arg]:
        arg = 2
    return arg


def decode_image(size, encodings):
    extraction_direction = argmin(size)
    extraction_slice = [slice(None), slice(None), slice(None)]
    image = np.empty(shape=size, dtype=np.uint8)
    for i, enc in enumerate(encodings):
        enc = np.asarray(bytearray(enc), np.uint8)
        image_slice = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        assert image_slice is not None, 'Cannot decode image'
        extraction_slice[extraction_direction] = i
        image[tuple(extraction_slice)] = image_slice
    return image


if __name__ == '__main__':
    # loading redis
    log = logging.getLogger("redis_bridge")
    log.setLevel(logging.INFO)
    r = redis.Redis(host='localhost', port=6379, db=0)
    print('Connected to Redis')
    
    # subscribing to channel
    pubsub = r.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe('ProstmorphBridge/compute_deformation/command')
    
    print('Waiting for commands')
    for command in pubsub.listen():  # blocking call
        if command['data'] == b'compute_deformation':
            print('Deforming MR onto US')
            try:
                # ---------------- RECEIVE AND DECODE IMAGE ----------------------------------
                mr_size = list(map(int, r.lrange('mr_size', 0, -1)))
                mr_origin = list(map(float, r.lrange('mr_origin', 0, -1)))
                mr_spacing = list(map(float, r.lrange('mr_spacing', 0, -1)))
                mr_direction = list(map(float, r.lrange('mr_direction', 0, -1)))
                mr_img_encodings = r.lrange('mr_image', 0, -1)
                mr_seg_encodings = r.lrange('mr_seg', 0, -1)
                us_size = list(map(int, r.lrange('us_size', 0, -1)))
                us_origin = list(map(float, r.lrange('us_origin', 0, -1)))
                us_spacing = list(map(float, r.lrange('us_spacing', 0, -1)))
                us_direction = list(map(float, r.lrange('us_direction', 0, -1)))
                us_img_encodings = r.lrange('us_image', 0, -1)
                us_seg_encodings = r.lrange('us_seg', 0, -1)
                
                assert None not in mr_img_encodings, 'Cannot read MR image from Redis database'
                assert None not in mr_seg_encodings, 'Cannot read MR segmentation from Redis database'
                assert mr_size, 'Cannot read MR size from Redis database'
                assert mr_origin, 'Cannot read MR origin from Redis database'
                assert mr_spacing, 'Cannot read MR spacing from Redis database'
                assert mr_direction, 'Cannot read MR direction from Redis database'
                assert None not in us_img_encodings, 'Cannot read US image from Redis database'
                assert None not in us_seg_encodings, 'Cannot read US segmentation from Redis database'
                assert us_size, 'Cannot read US size from Redis database'
                assert us_origin, 'Cannot read US origin from Redis database'
                assert us_spacing, 'Cannot read US spacing from Redis database'
                assert us_direction, 'Cannot read US direction from Redis database'
                
                # decode images (expected to be in RAS orientation)
                mr_img = decode_image(mr_size, mr_img_encodings)
                us_img = decode_image(us_size, us_img_encodings)
                mr_seg = decode_image(mr_size, mr_seg_encodings)
                us_seg = decode_image(us_size, us_seg_encodings)

                mr_img = np.transpose(mr_img, axes=(2, 1, 0))  # sitk expects dimensions in reverse order
                sitk_mr_img = sitk.GetImageFromArray(mr_img)
                sitk_mr_img.SetOrigin(mr_origin)
                sitk_mr_img.SetSpacing(mr_spacing)
                sitk_mr_img.SetDirection(mr_direction)
                
                mr_seg = np.transpose(mr_seg, axes=(2, 1, 0))
                sitk_mr_seg = sitk.GetImageFromArray(mr_seg)
                sitk_mr_seg.SetOrigin(mr_origin)
                sitk_mr_seg.SetSpacing(mr_spacing)
                sitk_mr_seg.SetDirection(mr_direction)

                us_img = np.transpose(us_img, axes=(2, 1, 0))
                sitk_us_img = sitk.GetImageFromArray(us_img)
                sitk_us_img.SetOrigin(us_origin)
                sitk_us_img.SetSpacing(us_spacing)
                sitk_us_img.SetDirection(us_direction)
                
                us_seg = np.transpose(us_seg, axes=(2, 1, 0))
                sitk_us_seg = sitk.GetImageFromArray(us_seg)
                sitk_us_seg.SetOrigin(us_origin)
                sitk_us_seg.SetSpacing(us_spacing)
                sitk_us_seg.SetDirection(us_direction)

                # ----------------- DEFORMATION ---------------------------------------------
                # prepare the input
                size, spacing = (160, 160, 160), (0.5, 0.5, 0.5)
                mr_img_res, mr_seg_res, _, _ = ImageLoader.prepare_sample(sitk_mr_img, sitk_mr_seg, spacing, size)
                us_img_res, _, _, _ = ImageLoader.prepare_sample(sitk_us_img, sitk_us_seg, spacing, size)
                # add batch and auxiliary dimensions
                mr_img_res = mr_img_res[None, ..., None]
                us_img_res = us_img_res[None, ..., None]
                mr_seg_res = mr_seg_res[None, ::2, ::2, ::2, None]
                test_input = [mr_img_res, us_img_res, mr_seg_res]
                # regress
                test_pred = model.predict(test_input)  # [def_mr_img, field, def_mr_seg]
                field = test_pred[1][0, ...]  # only 1 batch
                field_size = field.shape
                
                # ----------------- ENCODE FIELD AND SEND -------------------------------------
                extraction_direction = 2
                extraction_slice = [slice(None), slice(None), slice(None), slice(None, None, -1)]
                field_slice_encodings = []
                for i in range(field_size[extraction_direction]):
                    # itk::OpenCVBridge automatically casts BGR2RGB for 3-channel images, so we must send a BGR image, hence the ::-1
                    extraction_slice[extraction_direction] = i
                    success, field_slice_encoding = cv2.imencode('.tiff', field[tuple(extraction_slice)])
                    assert success, "Cannot encode field"
                    field_slice_encodings.append(field_slice_encoding.tobytes())
                    # print("(20, 20): ", field[20, 20, i, :])
                
                r.pipeline() \
                    .delete("field_size") \
                    .rpush("field_size", *field_size) \
                    .delete("field")\
                    .rpush("field", *field_slice_encodings)\
                    .execute()

                r.publish('ProstmorphBridge/compute_deformation/results', 'done')
                print('Field sent')
            except Exception as e:
                r.publish('ProstmorphBridge/compute_deformation/results', 'fail')
                print(e.msg)
        else:
            log.error('Unknown command received')

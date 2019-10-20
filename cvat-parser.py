import xml.etree.ElementTree as ET
import cv2

'''


into this csv, filename,x1, y1, x2, y2, class_name

'''

import os


def main():
    annot_path = './data/annotations'
    videos_path = './data/videos'
    images_path = './data/images'

    annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
    idx = 0
    for annot in annots:
        try:
            idx += 1

            et = ET.parse(annot)
            root = et.getroot()
            element_source = root.find('.//source')
            video_name = element_source.text
            video_name_simple = video_name.replace(' ', '-')

            stop_frame = root.find('.//stop_frame')

            element_tracks = root.findall('.//track')
            for track in element_tracks:
                warned = False
                class_name = track.attrib['label']
                track_id = track.attrib['id']

                print('label:', class_name)
                boxes = track.findall('box')

                for box in boxes:

                    box_att = box.attrib
                    frame = box_att['frame']
                    is_keyframe = box_att['keyframe']
                    x1 = box_att['xtl']
                    y1 = box_att['ytl']
                    x2 = box_att['xbr']
                    y2 = box_att['ybr']
                    outside = box_att['outside']

                    if is_keyframe == '1':
                        key_x1 = x1
                        key_y1 = y1
                        warned = False
                    else:
                        if (key_x1 == x1 and key_y1 == y1 and outside == '0'):
                            # to avoid lost tracked annotations. e.g. forgot to annotate during interpolation.
                            if not warned:
                                print('WARN: label:', class_name, ', track_id:', track_id,
                                      ', maybe lose tracking from keyframe: ', frame)
                                warned = True

                            continue

                    print('frame:', frame)
                    # pandas first then generate images?


        except Exception as e:
            print('error:', e)
            continue

    print('done parsing')


if __name__ == '__main__':
    main()

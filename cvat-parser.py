import xml.etree.ElementTree as ET
import cv2
import pandas as pd

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
    df = pd.DataFrame(columns=['filename', 'x1', 'y1', 'x2', 'y2', 'class_name', 'video_name', 'frame'])

    for annot in annots:
        print(' ')
        print('Processing annotation file:', annot)
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
                print('frames:')
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
                                print(' ')
                                print('WARN: label:', class_name, ', track_id:', track_id,
                                      ', maybe lose tracking from keyframe: ', frame)
                                warned = True

                            continue

                    print(' ', frame, end='')
                    image_name = makeImageName(frame, images_path, video_name_simple)
                    df = df.append({'filename': image_name,
                                    'x1': round(float(x1)),
                                    'y1': round(float(y1)),
                                    'x2': round(float(x2)),
                                    'y2': round(float(y2)),
                                    'class_name': class_name,
                                    'video_name': video_name,
                                    'frame': frame}, ignore_index=True)

                print(' ')  # end of frames

        except Exception as e:
            print('error:', e)
            continue

        df.to_csv('./training.csv', index=False)

        # every annotation xml file for its own video.
        print('Dump images:')
        video_frame = 0
        video = cv2.VideoCapture(os.path.join(videos_path, video_name))
        vdf = df[df['video_name'] == video_name]
        success, image = video.read()
        while (success):

            if sum(vdf['frame'] == str(video_frame)) > 0:
                image_name = makeImageName(str(video_frame), images_path, video_name_simple)
                cv2.imwrite(image_name, image)
                print(' ', image_name, end='')

            success, image = video.read()
            video_frame = video_frame + 1

        print(' ')  # Dump images:

    print('done parsing')


def makeImageName(frame, images_path, video_name_simple):
    return os.path.join(images_path, video_name_simple + "-F" + frame + ".png")


if __name__ == '__main__':
    main()

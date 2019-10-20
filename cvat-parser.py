import xml.etree.ElementTree as ET

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

            element_tracks = root.findall('.//track')
            for track in element_tracks:
                class_name = track.attrib['label']
                print('label:', class_name)
                for box in track.findall('box'):
                    box_att = box.attrib
                    frame = box_att['frame']
                    x1 = box_att['xtl']
                    y1 = box_att['ytl']
                    x2 = box_att['xbr']
                    y2 = box_att['ybr']
                    print('frame:', frame)


        except Exception as e:
            print('error:', e)
            continue



    print('done parsing')

if __name__ == '__main__':
    main()


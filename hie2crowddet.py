import json
import tqdm

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='metadata.json')
    parser.add_argument('--outfile',  default='output.odgt')

    args = parser.parse_args()

    global_img_idx = 0
    global_anno_idx = 0

    metadata_file = args.metadata
    with open(metadata_file) as f:
        metadata = json.load(f)

    output_json = ""

    for seq_idx in tqdm.tqdm(metadata.keys()):
        json_file = 'labels/train/track2&3/' + seq_idx + '.json'
        with open(json_file) as f:
            data = json.load(f)

        annolist = data['annolist']

        for i, anno in enumerate(annolist):

            temp_json = {}
            image_name = anno['image'][0]['name']
            anno_info = anno['annorect']

            temp_json['ID'] = "%02d%s" % (int(seq_idx), image_name[:-4])
            temp_json['gtboxes'] = []

            for j in range(len(anno_info)):
                x1, x2 = anno_info[j]['x1'][0], anno_info[j]['x2'][0]
                y1, y2 = anno_info[j]['y1'][0], anno_info[j]['y2'][0]

                temp_json['gtboxes'].append(
                    {
                        'fbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        'tag': 'person'
                    }
                )

                global_anno_idx += 1

            global_img_idx += 1

            temp_json = json.dumps(temp_json)
            output_json += temp_json
            output_json += '\n'

    with open(args.outfile, 'w') as outfile:
        outfile.write(output_json)


if __name__ == '__main__':
    main()
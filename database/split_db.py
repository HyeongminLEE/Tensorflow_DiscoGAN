import os
import argparse
import numpy as np
import dbread as db
import scipy.misc

parser = argparse.ArgumentParser(description='Data cutter')

# parameters
parser.add_argument('--input', type=str, default='./database')
parser.add_argument('--output', type=str, default='./output_split')


def main():
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    output_dir_A = output_dir + '/A'
    output_dir_B = output_dir + '/B'

    if not os.path.exists(output_dir_A):
        os.makedirs(output_dir_A)
    if not os.path.exists(output_dir_B):
        os.makedirs(output_dir_B)

    print('Reading the database...')
    database = db.DBreader(input_dir, batch_size=1, labeled=False, shuffle=False)
    print('Database load complete!!')

    total_batch = database.total_batch

    print('Generating...')
    for step in range(total_batch):
        print(str(step) + '/' + str(total_batch))
        img = database.next_batch()
        img_A = img[:, :, 0:256, :]
        img_B = img[:, :, 256:, :]

        img_A = img_A.reshape(256, 256, 3)
        img_B = img_B.reshape(256, 256, 3)

        scipy.misc.imsave(output_dir_A + '/' + str(step+1).zfill(6) + '.png', img_A)
        scipy.misc.imsave(output_dir_B + '/' + str(step+1).zfill(6) + '.png', img_B)

    print('finished!!')


if __name__ == "__main__":
    main()

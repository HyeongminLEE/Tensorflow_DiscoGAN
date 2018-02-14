import os
import argparse
import tensorflow as tf
import numpy as np
import dbread as db
from model import Discogan
import scipy.misc

parser = argparse.ArgumentParser(description='Easy Implementation of DiscoGAN')

# parameters
parser.add_argument('--train_A', type=str, default='./database_A')
parser.add_argument('--train_B', type=str, default='./database_B')
parser.add_argument('--out_dir', type=str, default='./output')
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=64)


def normalize(im):
    return im * (2.0 / 255.0) - 1


def denormalize(im):
    return (im + 1.) / 2.


# Function for save the generated result
def save_visualization(X, nh_nw, save_path='./vis/sample.jpg'):
    nh, nw = nh_nw
    h, w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh, w * nw, 3))

    for n, x in enumerate(X):
        j = int(n / nw)
        i = int(n % nw)
        img[j * h:j * h + h, i * w:i * w + w, :] = x

    scipy.misc.imsave(save_path, img)


def main():
    global_epoch = tf.Variable(0, trainable=False, name='global_step')
    global_epoch_increase = tf.assign(global_epoch, tf.add(global_epoch, 1))

    args = parser.parse_args()
    db_dir_A = args.train_A
    db_dir_B = args.train_B
    result_dir_AtoB = args.out_dir + '/result/AtoB'
    result_dir_BtoA = args.out_dir + '/result/BtoA'
    ckpt_dir = args.out_dir + '/checkpoint'

    if not os.path.exists(result_dir_AtoB):
        os.makedirs(result_dir_AtoB)
    if not os.path.exists(result_dir_BtoA):
        os.makedirs(result_dir_BtoA)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    total_epoch = args.epochs
    batch_size = args.batch_size
    vis_num = 4

    database_A = db.DBreader(db_dir_A, batch_size=batch_size, labeled=False, resize=[64, 64])
    db_for_vis_A = db.DBreader(db_dir_A, batch_size=vis_num, labeled=False, resize=[64, 64])
    database_B = db.DBreader(db_dir_B, batch_size=batch_size, labeled=False, resize=[64, 64])
    db_for_vis_B = db.DBreader(db_dir_B, batch_size=vis_num, labeled=False, resize=[64, 64])

    sess = tf.Session()
    model = Discogan(sess, batch_size)

    saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    total_batch_A = database_A.total_batch
    total_batch_B = database_B.total_batch

    if total_batch_A > total_batch_B:
        total_batch = total_batch_B
    else:
        total_batch = total_batch_A

    epoch = sess.run(global_epoch)
    while True:
        if epoch == total_epoch:
            break
        for step in range(total_batch):
            input_A = normalize(database_A.next_batch())
            input_B = normalize(database_B.next_batch())

            if step % 2 == 0:
                loss_D = model.train_discrim(input_A, input_B, epoch * total_batch + step)  # Train Discriminator and get the loss value
            loss_G = model.train_gen(input_A, input_B, epoch * total_batch + step)  # Train Generator and get the loss value

            if step % 100 == 0:
                print('Epoch: [', epoch, '/', total_epoch, '], ', 'Step: [', step, '/', total_batch, '], D_loss: ', loss_D, ', G_loss: ', loss_G)

            if step % 500 == 0:
                for_vis_A = normalize(db_for_vis_A.next_batch())
                for_vis_B = normalize(db_for_vis_B.next_batch())
                generated_samples_AB = denormalize(model.sample_generate(for_vis_A, 'AB', batch_size=4))
                generated_samples_ABA = denormalize(model.sample_generate(for_vis_A, 'ABA', batch_size=4))
                generated_samples_BA = denormalize(model.sample_generate(for_vis_B, 'BA', batch_size=4))
                generated_samples_BAB = denormalize(model.sample_generate(for_vis_B, 'BAB', batch_size=4))

                img_for_vis_AB = np.concatenate([denormalize(for_vis_A), generated_samples_AB, generated_samples_ABA], axis=2)
                img_for_vis_BA = np.concatenate([denormalize(for_vis_B), generated_samples_BA, generated_samples_BAB], axis=2)
                savepath_AB = result_dir_AtoB + '/output_' + 'EP' + str(epoch).zfill(3) + "_Batch" + str(step).zfill(6) + '.jpg'
                savepath_BA = result_dir_BtoA + '/output_' + 'EP' + str(epoch).zfill(3) + "_Batch" + str(step).zfill(6) + '.jpg'
                save_visualization(img_for_vis_AB, (vis_num, 1), save_path=savepath_AB)
                save_visualization(img_for_vis_BA, (vis_num, 1), save_path=savepath_BA)

        epoch = sess.run(global_epoch_increase)
        saver.save(sess, ckpt_dir + '/model_epoch' + str(epoch).zfill(3))


if __name__ == "__main__":
    main()

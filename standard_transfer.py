import configparser
import shutil
import time
from pprint import pprint as ppt

import tensorflow as tf

from psnet import PSNet
from utils import *

opj = os.path.join
ope = os.path.exists
om = os.mkdir
import glob

# os.environ['KMP_DUPLICATE_LIB_OK']='True' # Uncomment this line if you are using macOS

np.random.seed(42)
trained_model = "./trained_model/model"
st_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))
if not os.path.exists("style_transfer_results"):
    os.mkdir("style_transfer_results")
st_dir = os.path.join("style_transfer_results", st_time)
if not os.path.exists(st_dir):
    os.mkdir(st_dir)
shutil.copyfile("base_config.ini", os.path.join(st_dir, "config.ini"))
config = configparser.ConfigParser()
config.read(os.path.join(st_dir, "config.ini"))

# a list of path to content/style point clouds or images
CONTENT_LIST = glob.glob("./sample_content/*")
STYLE_LIST = glob.glob("./sample_style/*")
iteration = 10000

content_layer = list(map(lambda x: int(x), config["style_transfer"]["content_layer"].split(",")))
style_layer = list(map(lambda x: int(x), config["style_transfer"]["content_layer"].split(",")))
use_content_color = ["FE_COLOR_FE_{}".format(i) for i in content_layer]
use_style_color = ["FE_COLOR_FE_{}".format(i) for i in style_layer]

for idx, content_path in enumerate(CONTENT_LIST):
    content_dir = opj(st_dir, str(idx))
    if not ope(content_dir):
        om(content_dir)
    content_geo, content_ncolor = prepare_content_or_style(content_path)
    content_color = (127.5 * (content_ncolor + 1)).astype(np.int16)
    display_point(content_geo, content_color, fname=os.path.join(content_dir, "content.png"), axis="off", marker_size=3)

    # get content representations
    tf.reset_default_graph()
    sess = tf.Session()
    psnet = PSNet(config=config,
                  sess=sess,
                  train_dir=content_dir,
                  mode="test",
                  num_pts=content_ncolor.shape[0])
    psnet.restore_model(trained_model)
    ppt(list(psnet.node.keys()))
    # obtained_content_fvs = sess.run({i:psnet.node_color[i] for i in use_content_color},
    #                                 feed_dict={psnet.color: content_ncolor[None, ...],
    #                                            psnet.bn_pl: False,
    #                                            psnet.dropout_prob_pl: 1.0})
    obtained_content_fvs = sess.run(psnet.node, feed_dict={psnet.color: content_ncolor[None, ...],
                                                           psnet.geo: content_geo[None, ...],
                                                           psnet.bn_pl: False,
                                                           psnet.dropout_prob_pl: 1.0})
    sess.close()

    for st_idx, style_path in enumerate(STYLE_LIST):
        style_dir = opj(content_dir, str(st_idx))
        if not ope(style_dir):
            om(style_dir)

        if not (style_path.endswith("ply") or style_path.endswith("npy")):
            style_ncolor = prepare_content_or_style(style_path)
            from_image = True
        else:
            from_image = False
            style_geo, style_ncolor = prepare_content_or_style(style_path)
            style_color = (127.5 * (style_ncolor + 1)).astype(np.int16)
            display_point(style_geo, style_color, fname=os.path.join(style_dir, "style.png"), axis="off",
                          marker_size=3)

        # get style representations
        tf.reset_default_graph()
        sess = tf.Session()
        psnet = PSNet(config=config,
                      sess=sess,
                      train_dir=style_dir,
                      mode="test",
                      num_pts=style_ncolor.shape[0])
        psnet.restore_model(trained_model)
        if from_image:
            obtained_style_fvs = sess.run({i: psnet.node_color[i] for i in use_style_color},
                                          feed_dict={psnet.color: style_ncolor[None, ...],
                                                     psnet.bn_pl: False,
                                                     psnet.dropout_prob_pl: 1.0})
        else:
            obtained_style_fvs = sess.run(psnet.node,
                                          feed_dict={psnet.color: style_ncolor[None, ...],
                                                     psnet.geo: style_geo[None, ...],
                                                     psnet.bn_pl: False,
                                                     psnet.dropout_prob_pl: 1.0})
        obtained_style_fvs_gram = dict()
        for layer, fvs in obtained_style_fvs.items():
            gram = []
            for row in fvs:
                gram.append(np.matmul(row.T, row) / row.size)
            obtained_style_fvs_gram[layer] = np.array(gram)

        sess.close()

        tf.reset_default_graph()
        with tf.Graph().as_default() as graph:
            sess = tf.Session()
            psnet = PSNet(config=config,
                          sess=sess,
                          train_dir=style_dir,
                          mode="styletransfer",
                          num_pts=content_geo.shape[0],  # should be the same as content
                          target_content=obtained_content_fvs,
                          target_style=obtained_style_fvs_gram,
                          geo_init=tf.constant_initializer(value=content_geo),
                          color_init=tf.constant_initializer(value=content_ncolor),
                          from_image=from_image)
            psnet.restore_model(trained_model)
            previous_loss = float("inf")
            for i in range(iteration):
                psnet.style_transfer_one_step()
                if from_image:
                    current_total_loss = sess.run(psnet.total_loss_color, feed_dict={
                        psnet.bn_pl: False,
                        psnet.dropout_prob_pl: 1.0})
                else:
                    current_total_loss = sess.run(psnet.total_loss_color, feed_dict={
                        psnet.bn_pl: False,
                        psnet.dropout_prob_pl: 1.0}) + sess.run(psnet.total_loss_geo, feed_dict={
                        psnet.bn_pl: False,
                        psnet.dropout_prob_pl: 1.0})
                if abs(previous_loss - current_total_loss) < 1e-7 or i == iteration - 1:
                    transferred_color = (127.5 * (np.squeeze(np.clip(sess.run(psnet.color), -1, 1)) + 1)).astype(
                        np.int16)
                    if not from_image:
                        transferred_geo = np.squeeze(sess.run(psnet.geo))
                        save_ply(transferred_geo, transferred_color,
                                 os.path.join(style_dir, style_path.split("/")[-1].split(".")[0] + "_{}.ply".format(i)))
                        display_point(content_geo, transferred_color, fname=os.path.join(style_dir,
                                                                                         style_path.split("/")[
                                                                                             -1].split(".")[
                                                                                             0] + "_color_{}.png".format(
                                                                                             i)), axis="off",
                                      marker_size=2)
                        display_point(transferred_geo, transferred_color, fname=os.path.join(style_dir,
                                                                                             style_path.split("/")[
                                                                                                 -1].split(".")[
                                                                                                 0] + "_both_{}.png".format(
                                                                                                 i)), axis="off",
                                      marker_size=2)
                        display_point(transferred_geo, content_color, fname=os.path.join(style_dir,
                                                                                         style_path.split("/")[
                                                                                             -1].split(".")[
                                                                                             0] + "_geo_{}.png".format(
                                                                                             i)), axis="off",
                                      marker_size=2)
                    else:
                        save_ply(content_geo, transferred_color,
                                 os.path.join(style_dir, style_path.split("/")[-1].split(".")[0] + "_{}.ply".format(i)))
                        display_point(content_geo, transferred_color, fname=os.path.join(style_dir,
                                                                                         style_path.split("/")[
                                                                                             -1].split(".")[
                                                                                             0] + "_color_{}.png".format(
                                                                                             i)), axis="off",
                                      marker_size=2)

                    break
                previous_loss = current_total_loss
            sess.close()

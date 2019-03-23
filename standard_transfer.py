from pmnet import Pmnet
import configparser
import os
import tensorflow as tf
from pprint import pprint as ppt
import numpy as np
from utils import *
import time
import shutil
opj = os.path.join
ope = os.path.exists
om = os.mkdir
import glob

np.random.seed(42)
trained_model = "./trained_model/model"
st_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))
st_dir = os.path.join("style_transfer_results", st_time)
if not os.path.exists(st_dir):
    os.mkdir(st_dir)
shutil.copyfile("base_config.ini", os.path.join(st_dir, "config.ini"))
config = configparser.ConfigParser()
config.read(os.path.join(st_dir, "config.ini"))

CONTENT_PATH = ['/media/yang/367a94b6-e168-47b3-84d0-223f5dbc78e2/hoshino042/DensePoint/02958343_car,auto,automobile,machine,motorcar/ply_file/1f604bfb8fb95171ac94768c3754c895.ply']
# STYLE_PATH = ["image_style/Sunflowers_resized.jpg",
#               "image_style/starry_night_resized.jpg",
#               "image_style/hokusai_resized.jpg",
#               "image_style/colorful_resized.jpg",
#               "image_style/scream.jpg",
#               "image_style/water_resized.jpg"]
STYLE_PATH = glob.glob(os.path.join('/media/yang/367a94b6-e168-47b3-84d0-223f5dbc78e2/hoshino042/DensePoint/03636649_lamp/ply_file', "*.ply"))
iteration = 10000

content_layer = list(map(lambda x: int(x),config["style_transfer"]["content_layer"].split(",")))
style_layer = list(map(lambda x: int(x),config["style_transfer"]["content_layer"].split(",")))
use_content_color = ["FE_COLOR_FE_{}".format(i) for i in content_layer]
use_style_color = ["FE_COLOR_FE_{}".format(i) for i in style_layer]

for idx, content_path in enumerate(CONTENT_PATH):
    content_dir = opj(st_dir, str(idx))
    if not ope(content_dir):
        om(content_dir)
    content_geo,content_ncolor = prepare_content_or_style(content_path)
    content_color = (127.5 * (content_ncolor + 1)).astype(np.int16)
    display_point(content_geo, content_color, fname=os.path.join(content_dir, "content.png"), axis="off", marker_size=3)

    # get content representations
    tf.reset_default_graph()
    sess = tf.Session()
    pmnet = Pmnet(config=config,
                  sess=sess,
                  train_dir=content_dir,
                  mode="test",
                  num_pts=content_ncolor.shape[0])
    pmnet.restore_model(trained_model)
    ppt(list(pmnet.node.keys()))
    # obtained_content_fvs = sess.run({i:pmnet.node_color[i] for i in use_content_color},
    #                                 feed_dict={pmnet.color: content_ncolor[None, ...],
    #                                            pmnet.bn_pl: False,
    #                                            pmnet.dropout_prob_pl: 1.0})
    obtained_content_fvs = sess.run(pmnet.node,feed_dict={pmnet.color: content_ncolor[None, ...],
                                               pmnet.geo: content_geo[None,...],
                                               pmnet.bn_pl: False,
                                               pmnet.dropout_prob_pl: 1.0})
    sess.close()

    for st_idx, style_path in enumerate(STYLE_PATH):
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
        pmnet = Pmnet(config=config,
                      sess=sess,
                      train_dir=style_dir,
                      mode="test",
                      num_pts=style_ncolor.shape[0])
        pmnet.restore_model(trained_model)
        if from_image:
            obtained_style_fvs = sess.run({i:pmnet.node_color[i] for i in use_style_color},
                                      feed_dict={pmnet.color: style_ncolor[None, ...],
                                                 pmnet.bn_pl: False,
                                                 pmnet.dropout_prob_pl: 1.0})
        else:
            obtained_style_fvs = sess.run(pmnet.node,
                                          feed_dict={pmnet.color: style_ncolor[None, ...],
                                                     pmnet.geo: style_geo[None, ...],
                                                     pmnet.bn_pl: False,
                                                     pmnet.dropout_prob_pl: 1.0})
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
            pmnet = Pmnet(config=config,
                              sess=sess,
                              train_dir=style_dir,
                              mode="styletransfer",
                              num_pts=content_geo.shape[0], # should be the same as content
                              target_content=obtained_content_fvs,
                              target_style=obtained_style_fvs_gram,
                              geo_init= tf.constant_initializer(value=content_geo),
                              color_init=tf.constant_initializer(value=content_ncolor),
                              from_image=from_image)
            pmnet.restore_model(trained_model)
            previous_loss = 1000000
            for i in range(iteration) :
                pmnet.style_transfer_one_step()
                current_total_loss = sess.run(pmnet.total_loss_color, feed_dict={
                                                 pmnet.bn_pl: False,
                                                 pmnet.dropout_prob_pl: 1.0}) + sess.run(pmnet.total_loss_geo, feed_dict={
                                                 pmnet.bn_pl: False,
                                                 pmnet.dropout_prob_pl: 1.0})

                if abs(previous_loss - current_total_loss) < 1e-7:
                    clipped_pts_color = (127.5 * (np.squeeze(np.clip(sess.run(pmnet.color), -1, 1)) + 1)).astype(
                        np.int16)
                    transferred_geo = np.squeeze(sess.run(pmnet.geo))
                    save_ply(transferred_geo, clipped_pts_color,
                             os.path.join(style_dir, style_path.split("/")[-1].split(".")[0] + "_{}.ply".format(i)))
                    display_point(content_geo, clipped_pts_color, fname=os.path.join(style_dir,
                        style_path.split("/")[-1].split(".")[0] + "_color_{}.png".format(i)), axis="off",marker_size=2)
                    display_point(transferred_geo, clipped_pts_color, fname=os.path.join(style_dir,
                        style_path.split("/")[-1].split(".")[0] + "_both_{}.png".format(i)), axis="off",marker_size=2)
                    display_point(transferred_geo, content_color, fname=os.path.join(style_dir,
                        style_path.split("/")[-1].split(".")[0] + "_geo_{}.png".format(i)), axis="off",marker_size=2)

                    break
                previous_loss = current_total_loss
                if i == iteration - 1:
                    clipped_pts_color = (127.5 * (np.squeeze(np.clip(sess.run(pmnet.color), -1, 1)) + 1)).astype(np.int16)
                    save_ply(content_geo, clipped_pts_color, os.path.join(style_dir,style_path.split("/")[-1].split(".")[0]+"_{}.ply".format(i)))
                    display_point(content_geo, clipped_pts_color, fname=style_path.split("/")[-1].split(".")[0] + "_color_{}.png".format(i), axis="off",marker_size=2)
                    display_point(transferred_geo, clipped_pts_color, fname=style_path.split("/")[-1].split(".")[0] + "_both_{}.png".format(i), axis="off",marker_size=2)
                    display_point(transferred_geo, content_color, fname=style_path.split("/")[-1].split(".")[0] + "_geo_{}.png".format(i), axis="off",marker_size=2)
            sess.close()

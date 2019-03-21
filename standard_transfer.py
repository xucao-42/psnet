from pmnet import Pmnet
import configparser
import os
import tensorflow as tf
from pprint import pprint as ppt
import numpy as np
from utils import *
import time
import shutil

np.random.seed(42)
trained_model_dir = "../fenet_seperate_train_results/2018_12_04_20_15/"
st_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))
st_dir = os.path.join(trained_model_dir, "style_transfer", st_time)
if not os.path.exists(st_dir):
    os.mkdir(st_dir)
shutil.copyfile("base_config.ini", os.path.join(st_dir, "config.ini"))
config = configparser.ConfigParser()
config.read(os.path.join(st_dir, "config.ini"))
model_id = 40
model_dir = os.path.join(trained_model_dir, "model", "model-{}".format(model_id))

CONTENT_PATH = ['/media/yang/367a94b6-e168-47b3-84d0-223f5dbc78e2/hoshino042/S3D/Area_5/office_7/office_7.ply']
STYLE_PATH = ["image_style/Sunflowers_resized.jpg",
              "image_style/starry_night_resized.jpg",git 
              "image_style/hokusai_resized.jpg",
              "image_style/colorful_resized.jpg",
              "image_style/scream.jpg",
              "image_style/water_resized.jpg"]
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
    #display_point(content_geo, content_color, fname=os.path.join(content_dir, "content.png"), axis="off", marker_size=3)

    # get content representations
    tf.reset_default_graph()
    sess = tf.Session()
    pmnet = Pmnet(config=config,
                  sess=sess,
                  train_dir=content_dir,
                  mode="test",
                  num_pts=content_ncolor.shape[0])
    pmnet.restore_model(model_dir)
    ppt(list(pmnet.node.keys()))
    obtained_content_fvs = sess.run({i:pmnet.node_color[i] for i in use_content_color},
                                    feed_dict={pmnet.color: content_ncolor[None, ...],
                                               pmnet.bn_pl: False,
                                               pmnet.dropout_prob_pl: 1.0})
    # obtained_content_fvs = sess.run(pmnet.node,feed_dict={pmnet.color: content_ncolor[None, ...],
    #                                            pmnet.geo: content_geo[None,...],
    #                                            pmnet.bn_pl: False,
    #                                            pmnet.dropout_prob_pl: 1.0})
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
            display_point(style_geo, style_color, fname=os.path.join(content_dir, "style.png"), axis="off",
                      marker_size=3)

        # get style representations
        tf.reset_default_graph()
        sess = tf.Session()
        pmnet = Pmnet(config=config,
                      sess=sess,
                      train_dir=style_dir,
                      mode="test",
                      num_pts=style_ncolor.shape[0])
        pmnet.restore_model(model_dir)
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
            pmnet.restore_model(model_dir)
            previous_loss = 100000
            for i in range(iteration) :
                pmnet.style_transfer_one_step()
                current_total_loss = sess.run(pmnet.total_loss_color, feed_dict={
                                                 pmnet.bn_pl: False,
                                                 pmnet.dropout_prob_pl: 1.0})
                if abs(previous_loss - current_total_loss) < 1e-7:
                    clipped_pts_color = (127.5 * (np.squeeze(np.clip(sess.run(pmnet.color), -1, 1)) + 1)).astype(
                        np.int16)
                    save_ply(content_geo, clipped_pts_color,
                             os.path.join(style_dir, style_path.split("/")[-1].split(".")[0] + "_{}.ply".format(i)))
                    break
                previous_loss = current_total_loss
                if i == iteration - 1:
                    clipped_pts_color = (127.5 * (np.squeeze(np.clip(sess.run(pmnet.color), -1, 1)) + 1)).astype(np.int16)
                    save_ply(content_geo, clipped_pts_color, os.path.join(style_dir,style_path.split("/")[-1].split(".")[0]+"_{}.ply".format(i)))
                    # display_point(content_geo, clipped_pts_color, fname=os.path.join(style_dir, img.split(".")[0]+"_{}.png".format(i)), axis="off",marker_size=2)
            sess.close()

import tensorflow as tf

TFRecord_fp = 'resources/training_data/TFRecord_val.record'

# tf.train.Features(feature={'image/height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])), 'image/width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])), 'image/filename':tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])), 'image/source_id':tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])), 'image/encoded':tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])), 'image/format':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])), 'image/object/bbox/xmin':tf.train.Feature(float_list=tf.train.FloatList(value=[xmins])), 'image/object/bbox/ymin':tf.train.Feature(float_list=tf.train.FloatList(value=[ymins])), 'image/object/bbox/xmax':tf.train.Feature(float_list=tf.train.FloatList(value=[xmaxs])), 'image/object/bbox/ymax':tf.train.Feature(float_list=tf.train.FloatList(value=[ymaxs])), 'image/object/class/text':tf.train.Feature(bytes_list=tf.train.BytesList(value=[classes_text])), 'image/object/class/label':tf.train.Feature(int64_list=tf.train.Int64List(value=[classes])),})

for example in tf.python_io.tf_record_iterator(TFRecord_fp):
    result = tf.train.Example.FromString(example)
    print(result.features.feature['image/filename'].bytes_list.value)
    print(result.features.feature['image/object/bbox/xmin'].float_list.value)
    print(result.features.feature['image/object/bbox/ymin'].float_list.value)
    print(result.features.feature['image/object/bbox/xmax'].float_list.value)
    print(result.features.feature['image/object/bbox/ymax'].float_list.value)
    print(result.features.feature['image/object/class/label'].int64_list.value)
    print(result.features.feature['image/object/class/text'].bytes_list.value)
    input("Press Enter to continue...")

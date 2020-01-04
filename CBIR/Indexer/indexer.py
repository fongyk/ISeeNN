import sys
import os
from PIL import Image

project_root = '../CBIR'
sys.path.append('../')
sys.path.append(os.path.join(project_root, 'search_web'))

from feature_extractor import ResizeExtractor
from CBIR.image_server.models import DBImage, DBImageThumbnail, ImageServer
from models import Feature

import mongoengine

_MONGODB_USER = 'webclient' # your_username
_MONGODB_PASSWD = 'xxxxxxxxxxxxxxxx' # your_password
_MONGODB_HOST = '127.0.0.1' # the central server ip
_MONGODB_NAME = 'image_retrieval'
_MONGODB_DATABASE_HOST = \
    'mongodb://%s:%s@%s/%s' \
    % (_MONGODB_USER, _MONGODB_PASSWD, _MONGODB_HOST, _MONGODB_NAME)

mongoengine.connect(_MONGODB_NAME, host=_MONGODB_DATABASE_HOST)

extensions = {".jpg", ".JPG", ".jpeg"}

dataset = 'oxford5k'
dir_name = os.path.join('/data4/fong', dataset, 'images') # the directory you want to index

ext = ResizeExtractor((244, 244))
server_id = ImageServer.objects.get(server_name='Center').pk

for parent, dirnames, filenames in os.walk(dir_name):
    for filename in filenames:
        img_filename = os.path.join(parent, filename)
        filename, file_extension = os.path.splitext(img_filename)
        if file_extension not in extensions:
            continue

        print(img_filename)
        try:
            db_image = DBImage.objects.get(path=img_filename)
            image_id = db_image.pk
        except DBImage.DoesNotExist:
            im = Image.open(img_filename)
            if im.format != 'JPEG':
                continue
            db_image = DBImage(
                server=server_id,
                path=img_filename,
                width=im.width,
                height=im.height,
                mime_type='image/'+im.format.lower(),
                source=dataset
            )
            db_image.save()
            image_id = db_image.pk
        finally:
            try:
                feature = Feature.objects.get(image=image_id, identity='VGG19_resize')
            except Feature.DoesNotExist:
                try:
                    feat = ext.extract(img_filename)
                    Feature.objects(image=image_id,identity='VGG19_resize').update_one(
                        set__image=image_id,
                        set__dimension=feat.size,
                        set__model='VGG19',
                        set__data=feat.tobytes(),
                        upsert=True
                    )
                except:
                    pass

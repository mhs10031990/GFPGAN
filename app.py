import io
import json
import argparse
import cv2
import glob
import numpy as np
import os
from basicsr.utils import imwrite
import sys
from os import listdir
from os.path import isfile, join

from gfpgan import GFPGANer
from werkzeug.utils import secure_filename

import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template, redirect, url_for

import sys


UPLOAD_FOLDER = 'inputs/whole_imgs'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__,static_folder='results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods = ['GET', 'POST'])
def index():
    return render_template("index.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods = ['GET', 'POST'])
def upload_file():
    mydir = 'inputs/whole_imgs'
    filelist = [ f for f in os.listdir(mydir)]
    for f in filelist:
        os.remove(os.path.join(mydir, f))
    if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                print('No file part')
                return redirect(request.url)
            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                print('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return redirect(url_for('upload_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/main', methods=['POST','GET'])
def main():
    """Inference demo for GFPGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--upscale', type=int, default=2, help='The final upsampling scale of the image')
    parser.add_argument('--arch', type=str, default='clean', help='The GFPGAN architecture. Option: clean | original')
    parser.add_argument('--channel', type=int, default=2, help='Channel multiplier for large networks of StyleGAN2')
    parser.add_argument('--model_path', type=str, default='GFPGANCleanv1-NoCE-C2.pth')
    parser.add_argument('--bg_upsampler', type=str, default='realesrgan', help='background upsampler')
    parser.add_argument(
        '--bg_tile', type=int, default=400, help='Tile size for background sampler, 0 for no tile during testing')
    parser.add_argument('--test_path', type=str, default='upload/', help='Input folder')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument('--paste_back', action='store_false', help='Paste the restored faces back to images')
    parser.add_argument('--save_root', type=str, default='results', help='Path to save root')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')

    sys.argv = ['--model_path GFPGANCleanv1-NoCE-C2.pth --upscale 2 --test_path inputs/whole_imgs --save_root results --bg_upsampler realesrgan']

    args = parser.parse_args()
    if args.test_path.endswith('/'):
        args.test_path = args.test_path[:-1]
    os.makedirs(args.save_root, exist_ok=True)

    # background upsampler
    if args.bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is very slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from realesrgan import RealESRGANer
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None
    args.test_path = 'inputs/whole_imgs'
    # set up GFPGAN restorer
    restorer = GFPGANer(
        model_path=args.model_path,
        upscale=args.upscale,
        arch=args.arch,
        channel_multiplier=args.channel,
        bg_upsampler=bg_upsampler)
    img_list = sorted(glob.glob(os.path.join(args.test_path, '*')))
    print(img_list, '**')
    count = 4
    for img_path in img_list:
        count -= 1
        if count == 0:
            break
        # read image
        print('yes')
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img, has_aligned=args.aligned, only_center_face=args.only_center_face, paste_back=args.paste_back)

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            print('1')
            # save cropped face
            save_crop_path = os.path.join(args.save_root, 'cropped_faces', f'{basename}_{idx:02d}.png')
            imwrite(cropped_face, save_crop_path)
            # save restored face
            if args.suffix is not None:
                save_face_name = f'{basename}_{idx:02d}_{args.suffix}.png'
            else:
                save_face_name = f'{basename}_{idx:02d}.png'
            save_restore_path = os.path.join(args.save_root, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)
            # save comparison image
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, os.path.join(args.save_root, 'cmp', f'{basename}_{idx:02d}.png'))

        # save restored img
        if restored_img is not None:
            if args.ext == 'auto':
                extension = ext[1:]
            else:
                extension = args.ext

            if args.suffix is not None:
                save_restore_path = os.path.join(args.save_root, 'restored_imgs',
                                                 f'{basename}_{args.suffix}.{extension}')
            else:
                save_restore_path = os.path.join(args.save_root, 'restored_imgs', f'{basename}.{extension}')
            imwrite(restored_img, save_restore_path)

    onlyfiles = [f for f in listdir('results/restored_imgs') if isfile(join('results/restored_imgs', f))]
    onlyfiles.remove('.DS_Store')
    return render_template("index2.html", variable = onlyfiles[0])




if __name__ == '__main__':
    app.run(host="0.0.0.0")
    main()
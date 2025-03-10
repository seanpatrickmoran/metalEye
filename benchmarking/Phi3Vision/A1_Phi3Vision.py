import requests
import json

import numpy as np

import sqlite3
import sqlite_vec

from typing import List
import struct
import array
import datetime


import json
import math
import os
import re
import time
import glob
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from PIL import Image, ImageOps
from transformers import AutoTokenizer

from AlbersEmbed import _generate, _embedGenerate
from AlbersEmbed import *






import numpy as np
from PIL import Image
from io import BytesIO
import base64 



def arrayToPng(flat_array, width=65, height=65, vmax=255):
    npfArray = np.array(flat_array)
    normalized_array = np.zeros((width, height))
    for i in range(height):
        for j in range(width):
            value = npfArray[i*height+j] / vmax * 255
            if value > 255:
                normalized_array[i,j] = 255
            elif value < 0:
                normalized_array[i,j] = 0
            else:
                normalized_array[i,j] = round(value)

    uint8_array = normalized_array.astype(np.uint8)

    pil_image = Image.fromarray(uint8_array,mode='L')
    rgb_image = pil_image.convert('RGB')

    img_byte_arr = BytesIO()
    rgb_image.save(img_byte_arr, format='PNG')

    img_byte_arr.seek(0)  # Reset the pointer to the beginning of the byte stream

    png_image_from_bytes = Image.open(img_byte_arr)
    # name = f"/Users/sean/Documents/Master/Others/imageDoofus/aaa{str(np.random.randint(20000))}.png"
    # png_image_from_bytes.save(name)
    return png_image_from_bytes







def array_to_rgba_png(flat_array, width=65, height=65, vmax=255):
    npfArray = np.array(flat_array)
    rgba_image = np.zeros((width, height, 4))
    for i in range(height):
        for j in range(width):
            for k in range(4):
                if k==3:
                    rgba_image[i,j,k] = 255
                    continue
                value = npfArray[i*height+j] / vmax * 255
                if value > 255:
                    rgba_image[i,j,k] = 255
                else:
                    rgba_image[i,j,k] = round(value)
                # rgba_image[i*height+j+k] = npfArray[i*height+j] / vmax * 255
    # rgba_image = rgba_image.reshape((height, width, 4))  # Reshape to 65x65x1
    # print(rgba_image)

    # image = Image.fromarray(rgba_image, 'RGBA')


    pil_image = Image.fromarray(rgba_image, 'RGBA')
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format='PNG')

    # Get the PNG image as bytes (the image is now stored in memory)
    img_byte_arr.seek(0)  # Reset the pointer to the beginning of the byte stream

    # Convert the PNG bytes back into a PIL Image (this is now a PNG Image object in memory)
    png_image_from_bytes = Image.open(img_byte_arr)





    return png_image_from_bytes





    # buffer.seek(0)
    # png_image_data = buffer.read()
    # print(png_image_data)
    # image = Image.open(BytesIO(png_image_data))
    # image = Image.frombytes('PNG', (65,65), png_image_data, 'raw')
    # buffer.seek(0)
    # image_data = buffer.read()


    # pil_image = Image.fromarray(numpy_array)
    # img_byte_arr = BytesIO()
    # pil_image.save(img_byte_arr, format='PNG')

    # Get the PNG image as bytes (can be used further in the program without saving)
    # img_byte_arr.seek(0)
    # image_data = img_byte_arr.read()


    # image = Image.open(BytesIO(buffer.getvalue()))
    # image = Image.open(image_data)
    # return image
    # return png_image_data
    # return base64.b64encode(buffer.getvalue()).decode('utf-8')



def _apply_chat_template(prompt, images, verbose, apply_chat_template=True):
    if apply_chat_template is False:
        print(f'*** Prompt ***\n{prompt}\n*** Images ***\n{images}\n*** Output ***') if verbose else None
        return prompt, images
    if images is not None:
        images = [i for i in images] if isinstance(images, list) else [images]
        img_prompt = '\n'.join([f'<|image_{i+1}|>' for i in range(len(images))]) + '\n'
    else:
        img_prompt = ''
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [f"<|user|>\n{img_prompt}{i.strip()}<|end|>\n<|assistant|>\n" for i in prompt]
    if verbose:
        prompt_str = "\n".join(map(str.strip, prompt)).strip()
        images_str = "\n".join(map(str, images)) if images else "None"
        print(f'*** Prompt ***\n{prompt_str}\n*** Images ***\n{images_str}\n*** Output ***')
    prompt = prompt[0] if len(prompt) == 1 else prompt
    return prompt, images



def _apply_embed_template(prompt, images, verbose, apply_chat_template=True):
    if apply_chat_template is False:
        # print(f'*** Prompt ***\n{prompt}\n*** Images ***\n{images}\n*** Output ***') if verbose else None
        return prompt, images
    if images is not None:
        images = [i for i in images] if isinstance(images, list) else [images]
        img_prompt = '\n'.join([f'<|image_{i+1}|>' for i in range(len(images))]) + '\n'
    else:
        img_prompt = ''
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [f"<|user|>\n{img_prompt}{i.strip()}<|end|>\n<|assistant|>\n" for i in prompt]
    if verbose:
        prompt_str = "\n".join(map(str.strip, prompt)).strip()
        images_str = "\n".join(map(str, images)) if images else "None"
        # print(f'*** Prompt ***\n{prompt_str}\n*** Images ***\n{images_str}\n*** Output ***')
    prompt = prompt[0] if len(prompt) == 1 else prompt
    return prompt, images



def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)




def call(PATH,TIMEOUT):

    connection = sqlite3.connect(PATH, timeout=TIMEOUT)  # Set timeout to 10 seconds
    cursor = connection.cursor()
    return connection,cursor


def untouch(dbPATH,timeout):
    db = sqlite3.connect(dbPATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    try:
        db.execute("DROP TABLE vec_items")#, (kwargs['tablename'],))
#         print(cursor.fetchall())
        print(f"success")
    except Exception as e:
        print("connection failure")
        print(e)
    finally:
        db.commit()
        db.close()






def _createTable(dbPATH, timeout,**kwargs):

    # connection,cursor=call(dbPATH,timeout)
    try:
        db = sqlite3.connect(dbPATH)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        # cursor = connection.execute("CREATE TABLE vec_items USING vec0(embedding float[256])")
        db.execute("CREATE VIRTUAL TABLE vec_items USING vec0(key_id integer primary key, embedding float[3072])")
        # cursor = connection.execute("CREATE TABLE imag(name, dataset, condition, coordinates, numpyarr, viewing_vmax, dimensions, hic_path, PUB_ID, resolution, norm, meta)")
        print("table make; success")
        # db.close()

    except sqlite3.OperationalError as e:

        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print(message)
        print(e)
    finally:
        db.close()
        # cursor.close()
        # connection.close()


    # sqlite_vec.load(db)
def _readSOURCE_writeVECTOR(dbPATH1, dbPATH2,timeout,**kwargs):
# def _readMatchAllTable(dbPATH,timeout,**kwargs):
    def _readDB(offset, limit):
        connection_s,cursor_s=call(dbPATH1,timeout)
        # connection_t,cursor_t=call(dbPATH2,timeout)
        # dbvec = sqlite3.connect(dbPATH2)




        try:
            db = sqlite3.connect(dbPATH2)
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            cursor_s.row_factory = sqlite3.Row
            # cursor_s.execute("SELECT key_id, hist_rel, numpyarr FROM imag LIMIT ? OFFSET ?", (limit,offset))
            ### for bin16
            
            cursor_s.execute("SELECT key_id, hist_rel, numpyarr, epigenomicFactors, motifDirection, viewing_vmax FROM imag LIMIT ? OFFSET ?", (limit,offset))
            row_ids = []
            reply = []
            # llm = kwargs["model"]
            response = lambda: None
            response.embeddings = []
            incrementor = 0
            for en in cursor_s.fetchall():
                if incrementor%8!=0:
                    incrementor+=1
                    continue
                incrementor+=1

                try:
                    assert len(en[2])==4*65*65
                    row_ids += [en[0]]
                    rarr = b''

                    # harr = array.array('I', en[1])
                    # barr = array.array('f', en[2])
                    vMax = en[5]

                    #Takes PIL, don't base64 it. 
                    PNG = arrayToPng(array.array('f', en[2]),65,65,vMax)
                    print(PNG)


                    ### for epigenomics
                    # earr = array.array('f', en[3])

                    # for el in harr:
                    #     rarr += struct.pack('l', el)

                    # for i in range(65):
                    #     for j in range(65):
                    #         rarr += struct.pack('f',barr[i*65+j])
                            # if 28<i<38 and 28<j<38:
                            #     rarr += struct.pack('f',barr[i*65+j])
                    # ### for epigenomics
                    # for el in earr:
                    #     rarr += struct.pack('f', el)


                    # ### for motifs
                    # if en[4]!=":":
                    #     byteDir = bytes(en[4],'utf8')
                    #     for zbyte in byteDir:
                    #         rarr += struct.pack('I', zbyte)

                    blind_model=False
                    quantize_model=False
                    quantize_cache=False
                    use_adapter=False
                    max_tokens=512
                    verbose=True
                    return_tps=False
                    early_stop=False
                    stream=False
                    apply_chat_template=True
                    enable_api=False


                    prompt = "Center and bottom left corner are most important. Focus on sharp gradient changes."
                    embeddings = _embedGenerate(*kwargs["model"], *_apply_embed_template(prompt, PNG, verbose, apply_chat_template), max_tokens=max_tokens, verbose=verbose, return_tps=return_tps, early_stop=early_stop, stream=stream)
                    # _ , embeddings = _generate(*kwargs["model"], *_apply_chat_template(prompt, PNG, verbose, apply_chat_template), max_tokens=max_tokens, verbose=verbose, return_tps=return_tps, early_stop=early_stop, stream=stream)

                    itera = 0
                    
                    _vec = mx.zeros((embeddings[1].shape[2]),dtype=mx.bfloat16)
                    for i in range(1, len(embeddings)):
                        ilx = embeddings[i]
                        _vec += ilx.flatten()
                        itera+=1

                    _vec /= (len(embeddings)-1)
                    response.embeddings +=[_vec]
                    
                except Exception as e:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(e).__name__, e.args)
                    print(message)
                    print(e)


            print("dataloaded@@",end="")
            # print(len(response.embeddings))
            for idx,embd in enumerate(response.embeddings):
                # print(idx, type(idx),row_ids[idx], len(embd))
                db.execute("INSERT INTO vec_items(key_id, embedding) VALUES (?, ?)", [row_ids[idx], serialize_f32(embd[:])],)
            # print(f"success")



        except sqlite3.OperationalError as e:
            # the limiter overflows whatever is left.
            # just read everything left.
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # print([len(x) for x in reply])
            for x in range(len(reply)):
                if len(reply[x])!=16900:
                    print(row_ids[x])
                    continue    


            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            print(e)



        except Exception as e:
            print(e)

        finally:
            cursor_s.close()
            connection_s.close()
            db.commit()
            db.close()

    if not all(i in kwargs for i in ["limit","offset"]):
        raise Exception("need  \"limit\",\"offset\" in kwargs")
    return _readDB(kwargs['offset'],kwargs["limit"])


def mainProg():
    # dbSOURCE = "/Users/sean/Documents/Master/2025/Feb2025/sourceTables/database_14_bin.db"
    # dbSOURCE = "/Users/seanmoran/Documents/Master/2025/Feb2025/database_TEST/database_14_bin.db"
    dbSOURCE = "/Users/sean/Documents/Master/2025/Feb2025/sourceTables/TEST_database_24_1bin.db"

    # dbVECTOR = "/Users/sean/Documents/Master/2025/Feb2025/embeddedLoops/EB_databaseVEC_14.db"
    # dbVECTOR = "/Users/sean/Documents/Master/2025/Feb2025/testTables/metalLlamacppSPEEDTEST.db"
    # dbVECTOR = "debug.db"
    dbVECTOR = "/Users/sean/Documents/Master/2025/March2025/Phi3V/ebTable/A1_Phi3V.db"

    try:
        _createTable(dbVECTOR, 10)
    except sqlite3.OperationalError:
        print('huh.')
        untouch(dbVECTOR,100)
        _createTable(dbVECTOR, 10)

    print('continues')
    hardLimiter = 99130;

    insert_kwargs = {
        "limit": 256,
        "offset": 0,
        "entrypoint": "phi3Vision"
        }

    if insert_kwargs["entrypoint"]=="phi3Vision":
        # insert_kwargs["model"] = Llama(
        # model_path="/Users/sean/Downloads/llama-3.2-3b-instruct-q4_k_m.gguf",
        # n_gpu_layers=-1, # Uncomment to use GPU acceleration
        # # seed=1337, # Uncomment to set a specific seed
        # n_ctx=131072, # Uncomment to increase the context window
        # embedding=True,
        # verbose=False,
        # )


        model_path = "/Users/sean/Documents/safeTensors/Phi-3-vision-128k-instruct"
        with open(f"{model_path}/config.json", "r") as f:
            config = json.load(f)

        model_config = SimpleNamespace(**config)
        model = Phi3VForCausalLM(model_config)

        model_weight = [(k, v.transpose(0, 2, 3, 1) if "patch_embedding.weight" in k else v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]
        model.load_weights(model_weight)
        mx.eval(model.parameters())
        model.eval()

        processor = Phi3VProcessor(model_path)
        preload=model, processor

        insert_kwargs["model"] = preload




    while insert_kwargs["offset"] < hardLimiter:
        tnow = datetime.datetime.now()
        _readSOURCE_writeVECTOR(dbSOURCE, dbVECTOR, 10, **insert_kwargs)
        insert_kwargs["offset"] = insert_kwargs.get("offset", 0) + insert_kwargs.get("limit", 10)
        print(datetime.datetime.now() - tnow, datetime.datetime.now())
        print(insert_kwargs)


if __name__ == "__main__":
    now = datetime.datetime.now()
    mainProg();
    print("MLX: ", end="")
    print(datetime.datetime.now() - now)


"""
('Item size 2 for PEP 3118 buffer format string B does not match the dtype B item size 1.',)
"""




import sqlite_vec
# from termcolor import colored, cprint

from typing import List
import struct
import datetime
import sqlite3
import json

import mlx.core as mx
import numpy as np
import sys

# from scipy.stats import pearsonr

dbSOURCE = "/Users/seanmoran/Documents/Master/2025/Feb2025/database_TEST/database_17_bin.db"
# dbSOURCE = "/Users/seanmoran/Documents/Master/2025/Feb2025/database_TEST/database_14_bin.db"
dbVECTOR = "/Users/seanmoran/Documents/Master/2025/Feb2025/EB_databaseVEC_14.db"

#99129

colorMap = {
            5: '\x1b[95m',
            4: '\x1b[96m',
            3: '\x1b[92m',
            2:  '\x1b[39m',
            1:  '\x1b[93m',
            0:  '\x1b[91m\x1b[5m'
            }


store_answer = {
                # "imageScore":np.zeros(99129),
                # "epiScore":np.zeros(99129),
                # # "histogramScore":np.zeros(99129),
                # "p@k":np.zeros(99129),
                "imageScore": mx.array([0]*99129, dtype=mx.float32),
                "epiScore": mx.array([0]*99129, dtype=mx.float32),
                # "histogramScore":np.zeros(99129),
                "p@k": mx.array([0]*99129, dtype=mx.float32),
                }


# def pairwise_correlation(A, B):
#     am = A - np.mean(A, axis=0, keepdims=True)
#     bm = B - np.mean(B, axis=0, keepdims=True)
#     return am.T @ bm /  (np.sqrt(
#         np.sum(am**2, axis=0,
#                keepdims=True)).T * np.sqrt(
#         np.sum(bm**2, axis=0, keepdims=True)))

def mlx_pairwise_correlation(A, B):
    am = A - A.mean(axis=0)
    bm = B - B.mean(axis=0)
    # am = A - mx.mean(A, axis=0, keepdims=True)
    # bm = B - mx.mean(B, axis=0, keepdims=True)
    _numer = mx.matmul(am.T, bm, stream=mx.gpu)
    return _numer /  (mx.sqrt(
        mx.sum(am**2, axis=0,
               keepdims=True), stream=mx.gpu).T * mx.sqrt(
        mx.sum(bm**2, axis=0, keepdims=True), stream=mx.gpu))

mx_comp_pairwise = mx.compile(mlx_pairwise_correlation)

# class NumpyEncoder(json.JSONEncoder):
#     """ Special json encoder for numpy types """
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)



class MLXEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        # if isinstance(obj, mx.int8):
        #     return int(obj)
        # elif isinstance(obj, mx.float32):
        #     return float(obj)
        if isinstance(obj, mx.array):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)




def serialize_f32(vector: List[float]) -> bytes:
    return struct.pack("%sf" % len(vector), *vector)


def deserialize_f32(vector,size=3096):
    return struct.unpack(f"{size}f", vector)



def call(PATH,TIMEOUT):
    connection = sqlite3.connect(PATH, timeout=TIMEOUT)  # Set timeout to 10 seconds
    cursor = connection.cursor()
    return connection,cursor



def _readEmbeddingByKeyId(timeout, key_id=0):
    try:
        db = sqlite3.connect(dbVECTOR)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        x = db.execute("SELECT embedding FROM vec_items WHERE key_id = ? LIMIT 1", [str(key_id),]).fetchall()
        print(key_id, end=": ")
        db.close()
        return x.pop()

    except Exception as e:
        db.close()
        print("failure")
        print(e)
        return -1
        
        

def _readEmbeddingByKeyId(timeout, key_id=0):
    try:
        db = sqlite3.connect(dbVECTOR)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        x = db.execute("SELECT embedding FROM vec_items WHERE key_id = ? LIMIT 1", [str(key_id),]).fetchall()
        print(key_id, end=": ")
        db.close()
        return x.pop()

    except Exception as e:
        db.close()
        print("failure")
        print(e)
        return -1
        

def bruteforceKNN(id, eValue):
    print("@@@", end=" ")
    # print("@@@", end=" ", file=sys.stderr, flush=True)

    KNNTime = datetime.datetime.now()



    db = sqlite3.connect(dbVECTOR)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    query_Row = keyIdToRow(dbSOURCE,id, 10)

    if query_Row == None:
        print()
        return

    q_vmax  = query_Row[6]
    q_image = mx.array(query_Row[5], dtype=mx.float32) / q_vmax * 255
    q_image = q_image.astype(dtype=mx.int8)
    # q_image = mx.array([round(float(a)/q_vmax*255) for a in query_Row[5]])
    q_epigenomic = mx.array(query_Row[17],dtype=mx.float32)
    # print(len([int(x) for x in query_Row[8]]))
    # q_histogram = np.frombuffer(query_Row[8],dtype=int)

    rows = db.execute(
      """
        SELECT
          key_id,
          distance
        FROM vec_items
        WHERE embedding MATCH ?
        AND k = 8
      """,
    eValue,
    ).fetchall()
    print([x[0] for x in rows])
    # print([x[0] for x in rows], file=sys.stderr, flush=True)
    store_answer["p@k"][id] = sum(1-x[1] for x in rows)/len(rows)


    store_epiP = mx.array([0]*7, dtype=mx.float32)
    store_imgP = mx.array([0]*7, dtype=mx.float32)
    store_histP = mx.array([0]*7, dtype=mx.float32)

    # make a dummy matrix which is just the query image/epigenetics x 7
    # batch pearsons? make 3 matrices to hold all results. record indices to ignore.
    # pearson correlation all of it (to preserve compiled function). ignore specific indices.

    # pearsonTime = datetime.datetime.now()
    mxIndex,mxImgIndex,mxEpiIndex = 0,0,0
    while rows:
        node = rows.pop()
        # print(node[0])
        # print('he')
        # ~> Compare the Features of the query. use
        val = keyIdToRow(dbSOURCE, node[0], 10)

        if val is None or val == -2:
            print('\x1b[91m\x1b[5mMissed Result!!!\x1b[0m')
            pass
            # pairwise_correlation(i_vec, a_vec)
            #maximum penalty.
        elif val[0]==id:
            pass
        else:
            a_vmax  = val[6]
            # a_image = mx.array([round(float(a)/a_vmax*255) for a in val[5]])  ##store image size exact before writing to file.
            a_image = mx.array(val[5], dtype=mx.float32) / a_vmax * 255
            a_image = a_image.astype(dtype=mx.int8)
            a_epigenomic = mx.array(val[17],dtype=mx.float32)
            # a_histogram = np.frombuffer(val[8],dtype=int)
            try:
                epiP = mx_comp_pairwise(q_epigenomic, a_epigenomic)
                mxEpiIndex+=1
            except ValueError as e:
                print(f"ValueError: {e}")
                epi = mx.array(0,dtype=mx.float32)
            try:
                imgP = mx_comp_pairwise(q_image, a_image)
                mxImgIndex+=1
            except ValueError as e:
                print(f"ValueError: {e}")
                imgP = mx.array(0,dtype=mx.float32)
            # try:
            #     histP = pairwise_correlation(q_histogram, a_histogram)
            # except ValueError as e:
            #     print(f"ValueError: {e}")
            #     histP = 0
            # print(epiP.item(), imgP.item(), mxIndex, type(epiP))

            store_epiP[mxIndex] = epiP.item()
            store_imgP[mxIndex] = imgP.item()
            # store_histP += [histP]
            # print("epigenomic",epiP)
            # print("images", imgP)
            mxIndex += 1



    store_answer["epiScore"][id] = store_epiP.sum()/(mxImgIndex+1)
    store_answer["imageScore"][id] = store_imgP.sum()/(mxEpiIndex+1)
    # store_answer["histogramScore"][id] = sum(store_histP)/len(store_histP)
    # print("pearson: ", val[0], " ", (datetime.datetime.now() - pearsonTime)/(mxIndex+1))
    # store_answer["p@k"][id] = sum(x[1] for x in rows)/len(rows)
    logging=""
    for pname in ["epiScore", "imageScore", "p@k"]:
        logging +=  pname+": "
    # for pname in ["epiScore", "imageScore", "histogramScore", "p@k"]:
        # print(,end="")
        _store = (store_answer[pname][id].item() * 100 // 20)
        _store = 5 if store_answer[pname][id] > 0.94 else _store
        _store = _store if _store > 0 else 0

        # print(f"{colorMap[_store]}{store_answer[pname][id]}\x1b[0m",end=" ")
        logging += f"{colorMap[_store]}{store_answer[pname][id]}\x1b[0m" + " "
    logging += "\n"

    print(logging)
    print("KNN: ", datetime.datetime.now() - KNNTime)
    # print(logging, file=sys.stderr, flush=True)
    # print("epiScore: ",  store_answer["epiScore"][id], ", imageScore: ", store_answer["imageScore"][id] , ", p@k: ", store_answer["p@k"][id])




def keyIdToRow(dbPATH=dbSOURCE, key_id=1, timeout=10):
    if key_id==-1:
        return

    #Full DB, not VEC
    connection_s,cursor_s=call(dbPATH,timeout)

    try:
        cursor_s.row_factory = sqlite3.Row
        # print(key_id)
        cursor_s.execute("SELECT * FROM imag WHERE key_id = (?)", (key_id,))
        row = cursor_s.fetchone()
        cursor_s.close()
        connection_s.close()
        # print(row)
        return row

    except Exception as e:
        cursor_s.close()
        connection_s.close()
        print(e)
        return -2



def nameToKeyID(dbPATH=dbSOURCE, name="", timeout=10):
    print(name)
    if name=="":
        return

    #Full DB, not VEC
    connection_s,cursor_s=call(dbPATH,timeout)

    try:
        print(name)
        cursor_s.execute("SELECT key_id FROM imag WHERE name = ?", (name,))
        row = cursor_s.fetchone()
        cursor_s.close()
        connection_s.close()
        return row[0]

    except Exception as e:
        cursor_s.close()
        connection_s.close()
        print(e)
        return -2

def mainProg():

    fileOut = "022225_vector_pearson_analytics_Llama3.2:3B.json"
    for i in range(0,99114):
        embedded = _readEmbeddingByKeyId(10, i)
        # print(embedded)

        if embedded!=-1:
            # k_now = datetime.datetime.now()
            bruteforceKNN(i, embedded)
            # print(datetime.datetime.now() - k_now)

        if i%250==0:
            with open(fileOut, "w") as zug:
                zug.write(json.dumps(store_answer,cls=MLXEncoder))
    with open(fileOut, "w") as zug:
        zug.write(json.dumps(store_answer,cls=MLXEncoder))


if __name__ == "__main__":
    now = datetime.datetime.now()
    mainProg()
    print(datetime.datetime.now() - now)
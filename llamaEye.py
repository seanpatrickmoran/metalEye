# import requests
import json
# x = requests.post("http://127.0.0.1:8080/v1/chat/completions", headers = {"Content-Type": "application/json"},json={"messages": [{"role":"user", "content":"Katarina liked your song"}]} )







# from llama_cpp import Llama
import numpy as np




#testfire

import sqlite3
import sqlite_vec
from ollama import embed
import ollama
from typing import List
import struct
import array
import datetime


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
        db.execute("CREATE VIRTUAL TABLE vec_items USING vec0(key_id integer primary key, embedding float[512])")
        # cursor = connection.execute("CREATE TABLE imag(name, dataset, condition, coordinates, numpyarr, viewing_vmax, dimensions, hic_path, PUB_ID, resolution, norm, meta)")
        print("table make; success")
        # db.close()
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
            cursor_s.execute("SELECT key_id, hist_rel, numpyarr, epigenomicFactors, motifDirection FROM imag LIMIT ? OFFSET ?", (limit,offset))
            row_ids = []
            reply = []
            for en in cursor_s.fetchall():

                row_ids += [en[0]]
                rarr = b''

                harr = array.array('I', en[1])
                barr = array.array('f', en[2])
                ### for bin16
                earr = array.array('f', en[3])

                for el in harr:
                    rarr += struct.pack('l', el)

                for i in range(64):
                    for j in range(64):
                        # if i+1<=j and (i+j)%48==0:
                        #     rarr += struct.pack('f',barr[i+j])
                        if 28<i<38 and 28<j<38:
                            rarr += struct.pack('f',barr[i+j])

                ### for bin16
                for el in earr:
                    rarr += struct.pack('f', el)


                ### for bin16
                if en[4]!=":":
                    byteDir = bytes(en[4],'utf8')
                    for zbyte in byteDir:
                        rarr += struct.pack('I', zbyte)

                

                reply += [str(rarr)]
            print("dataloaded@@",end="")
            print(len(reply))

            if kwargs["entrypoint"]=="ollama":
                # pass
                response = ollama.embed(model='8KWin', input=reply, truncate=False, options={'num_gpus': 99})
                # response = ollama.embed(model='32Kllama', input=reply, truncate=False, options={'num_gpus': 99})
            elif kwargs["entrypoint"]=="llamacpp":
                pass
                # response = lambda: None
                # response.embeddings = []
                # llm = kwargs["model"]


                # embeddings = llm.create_embedding(reply)
                # _vec = np.zeros(len(embeddings['data'][0]['embedding'][0]))
                # for em in embeddings['data'][0]['embedding']:
                #     _vec[:] += np.array(em)[:]
                # _vec /= len(embeddings['data'][0]['embedding'])
                # response.embeddings +=[_vec]








                # print("NEED sequence level embeddings CHANGE HERE")
                # print([len(x) for x in embeddings])

                # response = lambda: None
                # response.embeddings = []
                # for rep in reply:
                #     print(rep)
                #     print(len(rep))
                    # _response = requests.post("http://127.0.0.1:8080/embedding", headers = {"Content-Type": "application/json"},json={"content": rep})
                    # print(len(json.loads(_response.text)[0]['embedding']))
                    # print([len(x) for x in json.loads(_response.text)[0]['embedding']])
                    # print(json.loads(_response.text)[0]['embedding'][0])
                    # print(len(json.loads(_response.text)[0]['embedding'][0]))
                    # print(jsonn.loads(_response.text)[0]['embedding'][0][:5])
                    # print(len(json.loads(_response.text)[1]))
                    # response.embeddings += [json.loads(_response.text)[1]['embedding']]
                    # print(len(response.embeddings))
                #     print(json.loads(_response.text))
                # _response = requests.post("http://127.0.0.1:8080/embedding", headers = {"Content-Type": "application/json"},json={"content": reply} )
                # print(_response.status_code)
                # print(len(json.loads(_response.text)[0]['embedding']))
                # response.embeddings = json.loads(_response.text)
            # response = ollama.embed(model='llama3.2', input=reply)
            print('write to file')
            # print(len(response.embeddings))
            # print(response.embeddings[0][0][:20])
            for idx,embd in enumerate(response.embeddings):
                db.execute("INSERT INTO vec_items(key_id, embedding) VALUES (?, ?)", [row_ids[idx], serialize_f32(embd[0:512])],)
            # print(f"success")



        except sqlite3.OperationalError as e:
            print(e)
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
    dbSOURCE = "/Users/sean/Documents/Master/2025/Feb2025/sourceTables/database_16_bin.db"

    # dbVECTOR = "/Users/sean/Documents/Master/2025/Feb2025/embeddedLoops/EB_databaseVEC_14.db"
    dbVECTOR = "/Users/sean/Documents/Master/2025/Feb2025/testTables/llamaSPEEDTEST.db"
    # dbVECTOR = "/Users/sean/Documents/Master/2025/Feb2025/embeddedLoops/EB_databaseVEC_16.db"

    try:
        _createTable(dbVECTOR, 10)
    except sqlite3.OperationalError:
        untouch(dbVECTOR,100)
        _createTable(dbVECTOR, 10)

    hardLimiter = 24;
    #check length of table

    insert_kwargs = {
        "limit": 8,
#        "offset": 1650,
        "offset": 0,
        # "entrypoint": "llamacpp"
        "entrypoint": "ollama"
        }

    if insert_kwargs["entrypoint"]=="llamacpp":
        pass
        # insert_kwargs["model"] = Llama(
        # model_path="/Users/seanmoran/Downloads/llama-3.2-3b-instruct-q8_0.gguf",
        # n_gpu_layers=-1, # Uncomment to use GPU acceleration
        # # seed=1337, # Uncomment to set a specific seed
        # n_ctx=8192, # Uncomment to increase the context window
        # embedding=True,
        # )

    while insert_kwargs["offset"] < hardLimiter:
        tnow = datetime.datetime.now()
        _readSOURCE_writeVECTOR(dbSOURCE, dbVECTOR, 10, **insert_kwargs)
        insert_kwargs["offset"] = insert_kwargs.get("offset", 0) + insert_kwargs.get("limit", 10)
        print(datetime.datetime.now() - tnow, datetime.datetime.now())
        print(insert_kwargs)
        # print("written")


if __name__ == "__main__":
    now = datetime.datetime.now()
    mainProg();
    print("ollama: ", end="")
    print(datetime.datetime.now() - now)







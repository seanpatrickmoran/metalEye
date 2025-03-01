import sqlite3
import sqlite_vec
import datetime
import json
import array

import mlx.core as mx


def call(PATH,TIMEOUT):
    connection = sqlite3.connect(PATH, timeout=TIMEOUT)  # Set timeout to 10 seconds
    cursor = connection.cursor()
    return connection,cursor

def measureSource(PATH, timeout):
    print(f"interfacing @ {PATH}")
    rowCount = 0
    connection_s,cursor_s=call(PATH,timeout)
    try:
        cursor_s.row_factory = sqlite3.Row
        cursor_s.execute("SELECT key_id FROM imag")

        for en in cursor_s.fetchall():
            rowCount += 1

    except sqlite3.OperationalError as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print(message)
        print(e)

    finally:
        cursor_s.close()
        connection_s.close()
        print(rowCount)
    return rowCount

def measureFTS(PATH, timeout):
    print(f"interfacing @ {PATH}")
    rowCount = 0
    connection_s,cursor_s=call(PATH,timeout)
    try:
        cursor_s.row_factory = sqlite3.Row
        cursor_s.execute("SELECT rowid FROM vector_table")

        for en in cursor_s.fetchall():
            rowCount += 1

    except sqlite3.OperationalError as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print(message)
        print(e)

    finally:
        cursor_s.close()
        connection_s.close()
        print(rowCount)
    return rowCount


def parition_key_ids_DB(PATH, timeout):
    print(f"interfacing @ {PATH}")
    connection_s,cursor_s=call(PATH,timeout)
    store_positions = {}
    lastHiC = "/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/GM12878/GM12878.hic"
    lastResolution = "2000"
    lastTool = "mustache"
    lastPosition = 0
    rowCount = 0
    try:
        cursor_s.row_factory = sqlite3.Row
        cursor_s.execute("SELECT hic_path,resolution,toolSource FROM imag")

        for en in cursor_s.fetchall():
            if (en[0]!=lastHiC or en[1]!=lastResolution or en[2]!=lastTool):
                store_positions[lastHiC+lastResolution+lastTool]=[lastPosition,rowCount-1] #rangeQuery excludes upper bound, [a,b), we will write inclusive.
                lastHiC = en[0]
                lastResolution = en[1]
                lastTool = en[2]
                lastPosition = rowCount
                print(en[0], last, rowCount)
            rowCount += 1
            last = en[0]


    except sqlite3.OperationalError as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print(message)
        print(e)

    finally:
        cursor_s.close()
        connection_s.close()
        store_positions[lastHiC+lastResolution+lastTool]=[lastPosition,rowCount-1]
        print(rowCount)

        for k,v in store_positions.items():
        	print(k, v)

    named = PATH.split("/")[-1].rstrip("_bin.db")
    print(named)
    with open(f"{named}_{datetime.datetime.now().isoformat(timespec='hours')}.json", "w") as zug:
        zug.write(json.dumps(store_positions))


def measureSqvec(PATH, timeout):
    print(f"interfacing @ {PATH}")
    db = sqlite3.connect(PATH)
    rowCount = 0
    try:
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        # cursor_s.row_factory = sqlite3.Row
        rows = db.execute("SELECT key_id from vec_items").fetchall()
        # cursor_s.execute("SELECT key_id, hist_rel, numpyarr FROM imag LIMIT ? OFFSET ?", (limit,offset))
        last = -8
        for en in rows:
            if en[0]-8 !=last:
                print(en[0], rowCount)
            last=en[0]

            rowCount += 1
    except sqlite3.OperationalError as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print(message)
        print(e)
    finally:
        db.close()
        print(rowCount)
    return rowCount








def checkSourceImages(PATH, timeout):
    print(f"interfacing @ {PATH}")
    rowCount = 0
    connection_s,cursor_s=call(PATH,timeout)
    try:
        cursor_s.row_factory = sqlite3.Row
        cursor_s.execute("SELECT key_id,numpyarr,dimensions FROM imag")

        for en in cursor_s.fetchall():
            fArr = array.array("f")
            if len(en[1])!= 16900:
                print(en[0],en[2])
                rowCount += 1

    except sqlite3.OperationalError as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print(message)
        print(e)

    finally:
        cursor_s.close()
        connection_s.close()
        print(rowCount)











def mainProg():
    dbSOURCE = "/Users/sean/Documents/Master/2025/Feb2025/sourceTables/database_17_bin.db"
    # /Users/sean/Documents/Master/2025/March2025/SqueakToy/ebTable/A1_llama3.2-3B-khImage+hist.db 
    # /Users/sean/Documents/Master/2025/March2025/SqueakToy/virtualTables/A1_llama3.2-3B-fts5_khImage+hist.db
    dbVECTOR = "/Users/sean/Documents/Master/2025/March2025/SqueakToy/ebTable/A1_llama3.2-3B-khImage+hist.db"
    dbFtsVector = "/Users/sean/Documents/Master/2025/March2025/SqueakToy/virtualTables/A1_llama3.2-3B-fts5_khImage+hist.db"

    dbs = measureSource(dbSOURCE, 10)
    checkSourceImages(dbSOURCE,10)
    dbv = measureSqvec(dbVECTOR, 10)
    dbv_fts = measureFTS(dbFtsVector, 10)

    print(dbs, dbv, dbv_fts)
    # parition_key_ids_DB(dbSOURCE,10)

if __name__ == "__main__":
    # now = datetime.datetime.now()
    mainProg();
    # print("MLX: ", end="")
    # print(datetime.datetime.now() - now)











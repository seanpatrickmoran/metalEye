import mlx.core as mx
import json
import numpy as np
import time
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
path = "/Users/sean/Documents/Master/2025/Feb2025/table_18_metadata/LLama/Image+Hist_Only/022525_faissIVFPQ_vector_linear_pearson_analytics.json_2"
# path = "/Users/seanmoran/Projects/MLX/metalEye/utils/022425_faissIVFPQ_vector_pearson_analytics.fuse_2_3.json"
with open(path) as json_file:
	data = json.load(json_file)


path2 = "/Users/sean/Documents/Master/2025/Feb2025/table_18_metadata/LLama/Image+Hist_Only/022525_faissIVFPQ_vector_linear_pearson_analytics.json_3"
with open(path2) as json_file:
	data2 = json.load(json_file)

images1 = mx.array(data['imageScore'])[:]
epi1 = mx.array(data['epiScore'])[:]
pk1 = mx.array(data['p@k'])[:]

images_a = mx.nan_to_num(images1)

seam = 0
tolerance = 25
for i,n in enumerate(images_a):
	print(i,n.item())
	if n.item()==0.0:
		print(i,n, tolerance)
		# time.sleep(1)
		tolerance -= 1
	else:
		tolerance = 25
	if tolerance < 1:
		print(i,n,i-25)
		seam = i-25
		break

images2 = mx.array(data2['imageScore'])[:]
epi2 = mx.array(data2['epiScore'])[:]
pk2 = mx.array(data2['p@k'])[:]

for i in range(len(store_answer["imageScore"])):
	print(i, end=", ")
	if i < seam:
		print(images1[i].item())
		store_answer["imageScore"][i] = images1[i].item()
		store_answer["epiScore"][i] = epi1[i].item()
		store_answer["p@k"][i] = pk1[i].item()
	else:
		print(images2[i].item())
		store_answer["imageScore"][i] = images2[i].item()
		store_answer["epiScore"][i] = epi2[i].item()
		store_answer["p@k"][i] = pk2[i].item()


print("fused!!!")
time.sleep(5)


images0 = mx.array(store_answer["imageScore"])
images0a = mx.nan_to_num(images0)
seam = 0
tolerance = 25
for i,n in enumerate(images0a):
	print(i,n.item())
	if n.item()==0:
		tolerance -= 1
	else:
		tolerance = 1000
	if tolerance < 1:
		print(i,n,i-25)
		seam = i-25
		break



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

with open("/Users/sean/Documents/Master/2025/Feb2025/table_18_metadata/LLama/Image+Hist_Only/Llama3_Hist+Image.fuse_2_3.json", "w") as zug:
    zug.write(json.dumps(store_answer,cls=MLXEncoder))
import matplotlib.pyplot as plt 
import matplotlib
import mlx.core as mx
import json
import numpy as np


matplotlib.rcParams.update({'errorbar.capsize': 2})

#llama hist + image only #v17 source v18 embeddings
path = "/Users/sean/Documents/Master/2025/Feb2025/table_18_metadata/LLama/Image+Hist_Only/s17_v14//DatasetAll/022125_vector_pearson_analytics.json"
with open(path) as json_file:
	data = json.load(json_file)



images = mx.array(data['imageScore'])[:99113]
images_a = mx.nan_to_num(images)
image_VAL = images_a.mean()
images_a.std()


epigen = mx.array(data['epiScore'])[:99113]
epigen_a = mx.nan_to_num(epigen)
epigen_VAL = epigen_a.mean()
epigen_a.std()/mx.sqrt(epigen_a.shape[0])
epigen_a.std()

a =["ImageScore","EpiScore"]
X_axis = np.arange(len(a)) 

b =[image_VAL.item(), epigen_VAL.item()]
c =[(images_a.std()/mx.sqrt(images_a.shape[0])).item(), (epigen_a.std()/mx.sqrt(epigen_a.shape[0])).item()]
plt.bar(X_axis - 0.2, b,  0.2, label = 'Llama All Datasets')
plt.errorbar(X_axis - 0.2, b, yerr=c, ecolor='r', ls='none')

# path2 = "/Users/seanmoran/Projects/MLX/metalEye/utils/022425_faissIVFPQ_vector_pearson_analytics.FUSED.json"
# path2 = "/Users/seanmoran/Desktop/022425_faissIVFPQ_vector_pearson_analytics.json"


#llama hist + image only (IdSelectorRange)
path2 ="/Users/sean/Documents/Master/2025/Feb2025/table_18_metadata/LLama/Image+Hist_Only/s17_v14/RangeBound/022425_faissIVFPQ_vector_pearson_analytics.fuse_2_3_4.json"
with open(path2) as json_file:
	data2 = json.load(json_file)



images2 = mx.array(data2['imageScore'])[:99113]
images2_a = mx.nan_to_num(images2)
image2_VAL = images2_a.mean()
images2_a.std()


epigen2 = mx.array(data2['epiScore'])[:99113]
epigen2_a = mx.nan_to_num(epigen2)
epigen2_VAL = epigen2_a.mean()
epigen2_a.std()

d =["Image Similarity","Epigenetic Similarity"]
e =[image2_VAL.item(), epigen2_VAL.item()]
f =[(images2_a.std()/mx.sqrt(images2_a.shape[0])).item(), (epigen2_a.std()/mx.sqrt(epigen2_a.shape[0])).item()]
# plt.bar(d, e)
plt.bar(X_axis , e,  0.2, label = 'Llama Intra-dataset')
plt.errorbar(X_axis , e, yerr=f, ecolor='r', ls='none')


#llama hist + image + epi #v17 source v18 embeddings
path = "/Users/sean/Documents/Master/2025/Feb2025/table_18_metadata/Llava/Image+Hist+Epi/s17_v18(17)/DatasetAll/022625_Llava_Hist+Image.fuse_2_3.json"
with open(path) as json_file:
	data3 = json.load(json_file)

images3 = mx.array(data3['imageScore'])[:99113]
images3_a = mx.nan_to_num(images3)
image3_VAL = images3_a.mean()


epigen3 = mx.array(data3['epiScore'])[:99113]
epigen3_a = mx.nan_to_num(epigen3)
epigen3_VAL = epigen3_a.mean()

g =["Image Similarity","Epigenetic Similarity"]
h =[image3_VAL.item(), epigen3_VAL.item()]
i =[(images3_a.std()/mx.sqrt(images3_a.shape[0])).item(), (epigen3_a.std()/mx.sqrt(epigen3_a.shape[0])).item()]
# plt.bar(d, e)
plt.bar(X_axis + 0.2, h,  0.2, label = 'Llava All Datasets')
plt.errorbar(X_axis + 0.2, h, yerr=i, ecolor='r', ls='none')



plt.xticks(X_axis, a) 
plt.xlabel("Scoring Type") 
plt.ylabel("Pearson Score") 
plt.title("Sim Search Performance in Search Spaces") 
plt.legend() 

print("images::")
print("mean: ", end="")
print(image_VAL,image2_VAL,image3_VAL)
print("se: ", end="")
print((images_a.std()/mx.sqrt(images_a.shape[0])).item(),(images2_a.std()/mx.sqrt(images2_a.shape[0])).item(),(images3_a.std()/mx.sqrt(images3_a.shape[0])).item())
print()
print()

print("epigenomics::")
print("mean: ", end="")
print(epigen_VAL,epigen2_VAL,epigen3_VAL)
print("se: ", end="")
print((epigen_a.std()/mx.sqrt(epigen_a.shape[0])).item(),(epigen2_a.std()/mx.sqrt(epigen2_a.shape[0])).item(),(epigen3_a.std()/mx.sqrt(epigen3_a.shape[0])).item())
print()



plt.show()






# import numpy as np  
# import matplotlib.pyplot as plt  
  
# X = ['Group A','Group B','Group C','Group D'] 
# Ygirls = [10,20,20,40] 
# Zboys = [20,30,25,30] 
  
# X_axis = np.arange(len(X)) 
  
# plt.bar(X_axis - 0.2, Ygirls, 0.4, label = 'Girls') 
# plt.bar(X_axis + 0.2, Zboys, 0.4, label = 'Boys') 
  
# plt.xticks(X_axis, X) 
# plt.xlabel("Groups") 
# plt.ylabel("Number of Students") 
# plt.title("Number of Students in each group") 
# plt.legend() 
# plt.show() 

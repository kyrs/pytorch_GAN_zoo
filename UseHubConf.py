from hubconf import PGAN
import matplotlib.pyplot as plt
import torchvision
import torch
import numpy as np 

class PaGanSampler():
	def __init__(self):
		self.model = PGAN()
		self.decoderNoiseSz  = 512
	def sampleNoise(self,count):
		## function to sample noise from the model
		inputRandom, randomLabels = self.model.buildNoiseData(count)
		return inputRandom
	def sampleNormalizedImage(self,noise,npType=False):
		## generate normalized image for give noise
		if npType:
			noise= torch.Tensor(noise)

		out = self.model.test(noise,
	           getAvG=True,
	    
	           toCPU=True)
		return out.numpy()	

	def SampleImage(self,noise,npType=False,image=False,pathImage=None):
		## returen image in range 0-1
		if npType:
			noise=torch.Tensor(noise)
		out = self.model.test(noise,
	           getAvG=True,
	    
	           toCPU=True)
		# grid = torchvision.utils.make_grid(out.clamp(min=-1, max=1), scale_each=True, normalize=True)
		# return grid.numpy()
		if not image:
			return out.numpy()
		else :
			torchvision.utils.save_image(out.clamp(min=-1, max=1),pathImage,scale_each=True, normalize=True)
			return

	def SeeImage(self,noise,npType=False):
		## returen image in range 0-1
		if npType:
			noise=torch.Tensor(noise)

		out = self.model.test(noise,
	           getAvG=True,
	    
	           toCPU=True)
		# grid = torchvision.utils.make_grid(out.clamp(min=-1, max=1), scale_each=True, normalize=True)
		# return grid.numpy()

		gridOut = torchvision.utils.make_grid(out.clamp(min=-1, max=1),scale_each=True, normalize=True)
		plt.imshow(gridOut.permute(1, 2, 0).cpu().numpy())
		plt.show() 
if __name__ =="__main__":
	
	################### genereate sample to save images ################
	# sampNum =100
	# Obj = PaGanSampler()

	# listToSave = []
	# noise = Obj.sampleNoise(sampNum)
	# print(noise.size())
	# newIdx = np.split(noise,sampNum)
	
	# for ns in newIdx:
	# 	Obj.SeeImage(ns)
	# 	item = input("is it Ok ? press (y/n)")
	# 	if item=='y':
	# 		listToSave.append(ns.numpy())
	# 		print("added")
	# 		print(len(listToSave))
	# 	else:
	# 		pass 
	# 	if item=='q':
	# 		break
	# np.save('testSampleForAttr.npy', listToSave)

	fileName = "testSampleForAttr.npy"
	loadedNpFile = np.load(fileName)
	Obj = PaGanSampler()
	for elm in loadedNpFile:
		Obj.SeeImage(elm,npType=True)

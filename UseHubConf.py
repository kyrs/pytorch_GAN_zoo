from hubconf import PGAN
import matplotlib.pyplot as plt
import torchvision
import torch

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

	def SampleImage(self,noise,npType=False):
		## returen image in range 0-1
		if npType:
			noise=torch.Tensor(noise)
		out = self.model.test(noise,
	           getAvG=True,
	    
	           toCPU=True)
		# grid = torchvision.utils.make_grid(out.clamp(min=-1, max=1), scale_each=True, normalize=True)
		# return grid.numpy()
		return out.numpy()
if __name__ =="__main__":
	Obj = PaGanSampler()
	noise = Obj.sampleNoise(16)
	print(type(noise))
	# print(Obj.SampleImage(noise).numpy())
from hubconf import PGAN
import matplotlib.pyplot as plt
import torchvision


class PaGanSampler():
	def __init__(self):
		self.model = PGAN()
		self.decoderNoiseSz  = 512
	def sampleNoise(self,count):
		## function to sample noise from the model
		inputRandom, randomLabels = self.model.buildNoiseData(count)
		return inputRandom
	def sampleNormalizedImage(self,noise):
		## generate normalized image for give noise
		out = model.test(inputRandom,
	           getAvG=True,
	    
	           toCPU=True)
		return out	

	def SampleImage(self,noise):
		## returen image in range 0-1
		out = model.test(noise,
	           getAvG=True,
	    
	           toCPU=True)
		grid = torchvision.utils.make_grid(out.clamp(min=-1, max=1), scale_each=True, normalize=True)
		return grid.numpy()

if __name__ =="__main__":
	Obj = PaGanSampler()
	(Obj.sampleNoise(1))
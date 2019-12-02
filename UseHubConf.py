from hubconf import PGAN
import matplotlib.pyplot as plt
import torchvision


if __name__ =="__main__":
	model = PGAN()
	inputRandom, randomLabels = model.buildNoiseData(2)
	print(inputRandom.shape)
	out = model.test(inputRandom,
           getAvG=True,
    
           toCPU=True)
	
	grid = torchvision.utils.make_grid(out.clamp(min=-1, max=1), scale_each=True, normalize=True)
	print(grid.numpy())
	#plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
	# plt.show()